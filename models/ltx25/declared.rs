//! The standalone LTX binding: model metadata/math on one owner lane, with
//! all readers, dependencies, readiness and joins owned by Rust core.
use crate::{cuda::Device,frontend::GemmaTokenizer,graph,linear::Linear,startup,weights::Checkpoint};
use native_core_infra::{checkpoint,declared_startup::{AtUse,Bindings,Component,ConsumerGuard,Declaration,Preparation,Result},
    immutable_assets::ImmutableAssets,model_load::Options,startup_native::{LoadMode,NativeLoadPlan,NativeLoads,WeightStage},
    startup_owner::{OwnerBindings,OwnerExecutor},startup_policy::CpuPrepared};
use serde_json::{json,Value};
use std::{cell::RefCell,collections::{BTreeMap,BTreeSet},path::PathBuf,rc::Rc,sync::{Arc,Mutex},time::Instant};

const STAGES:[&str;5]=["text","connectors","dit","vae","upscale"];
#[derive(Clone)]
pub(crate) struct Resources {
    pub device:Rc<Device>,pub linear:Rc<Linear>,pub assets:Rc<ImmutableAssets>,pub linear_shapes:usize,
}
struct Target {key:String,text_root:PathBuf,checkpoints:Vec<(&'static str,Rc<Checkpoint>)>}
pub(crate) struct LtxStartup {
    runtime:OwnerExecutor<'static>,resources:Rc<RefCell<Option<Resources>>>,
    target:Rc<RefCell<Option<Target>>>,pub attention:Rc<RefCell<BTreeMap<PathBuf,usize>>>,
    loads:NativeLoads,tokenizer:Arc<Mutex<Option<CpuPrepared<GemmaTokenizer>>>>,
    installed:Rc<RefCell<BTreeSet<String>>>,
    key:String,generation:u64,retained_budget:usize,
}
fn required(name:&str)->Vec<String> {
    let mut value=vec!["resident-weights".into(),format!("install-{name}")];
    if name=="text" {value.push("model-tokenizer".into());value.push("text-backend".into());}
    if name=="connectors" {value.push("text-backend".into());}
    value
}
fn prep(name:&str,before:&[&str],weights:&[&str],after:&[&str],cpu:bool,request:bool,at_use:AtUse)->Preparation {
    Preparation {name:name.into(),before:before.iter().map(|v|v.to_string()).collect(),weights:weights.iter().map(|v|v.to_string()).collect(),
        after:after.iter().map(|v|v.to_string()).collect(),inputs:if request{vec!["request".into()]}else{vec![]},cpu,at_use,initialize:false,serial_group:None}
}
fn target_key(inputs:&Value)->Result<&str>{inputs["request"].as_str().ok_or_else(||"LTX request key missing".into())}
fn scope_target<'a>(value:&'a Option<Target>,key:&str)->Result<&'a Target>{value.as_ref().filter(|v|v.key==key).ok_or_else(||"LTX startup profile belongs to another scope".into())}
fn ready_tensors(names:&[String],target:&Target,device:&Device)->bool {
    names.iter().all(|name|name.split_once('.').and_then(|(prefix,key)| {
        let checkpoint=&target.checkpoints.iter().find(|(p,_)|*p==prefix)?.1;
        device.cached_weight(&format!("{}\0{key}",checkpoint.path.display()))
    }).is_some())
}
impl LtxStartup {
    pub fn new(checkpoint:Arc<checkpoint::Declaration>,activate:bool)->Result<Self> {
        let options=Options::for_declaration(&checkpoint)?;
        // Preserve the original bounded cold reader/upload policy. A resident
        // owner prepares its reusable host cache after sampling has started.
        let groups=startup::stages(&checkpoint).map_err(|e|e.to_string())?;
        let names=groups.iter().map(|g|g.iter().map(|r|r.name.clone()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let stages=STAGES.into_iter().zip(groups).map(|(name,tensors)|WeightStage {
            component:Component {name:name.into(),after:vec![],required_preparations:Some(required(name)),reader_limit_mib:0,prefetch:false,load_at_use:false,initialize:false},tensors}).collect();
        let mode=if activate {LoadMode::Independent}else{LoadMode::IndependentAtUse};
        let plan=NativeLoadPlan::new(checkpoint.manifest.clone(),checkpoint.roots.clone(),options,mode,stages)?;
        let mut shared=Bindings::default();let mut local=OwnerBindings::default();
        let resources=Rc::new(RefCell::new(None));let owner=resources.clone();
        let device_checkpoint=checkpoint.clone();
        let loads=unsafe {plan.bind(&mut shared,&mut local,move || {
            let assets=Rc::new(ImmutableAssets::new(2usize<<30,4096));
            let mut device=Device::open_private(0).map_err(|e|e.to_string())?;device.host_assets=Some(assets.clone());
            let linear=Linear::new(&device).map_err(|e|e.to_string())?;
            let shapes=startup::linear_shapes(&device_checkpoint).map_err(|e|e.to_string())?;
            native_core_infra::startup_trace::measure("startup_backend_prepare","linear metadata",0,||linear.prepare(&shapes)).map_err(|e|e.to_string())?;
            let stream=device.stream as usize;
            *owner.borrow_mut()=Some(Resources {device:Rc::new(device),linear:Rc::new(linear),assets,linear_shapes:shapes.len()});
            Ok(stream)
        })}?;
        let tokenizer=Arc::new(Mutex::new(None::<CpuPrepared<GemmaTokenizer>>));
        let slot=tokenizer.clone();
        shared.insert("cpu","model-tokenizer",Arc::new(move |_| {
            let begun=Instant::now();let value=startup::tokenizer(&checkpoint).map_err(|e|e.to_string())?;
            *slot.lock().unwrap()=Some(CpuPrepared {value,seconds:begun.elapsed().as_secs_f64()});
            Ok(json!({"tokenizer_decoded":true}))
        }));
        let slot=tokenizer.clone();
        shared.insert("validate_cpu","model-tokenizer",Arc::new(move |v|Ok(json!(v["tokenizer_decoded"]==true && slot.lock().unwrap().is_some()))));
        shared.insert("run","model-tokenizer",Arc::new(|inputs|Ok(inputs["prepared"].clone())));
        shared.insert("validate_run","model-tokenizer",Arc::new(|v|Ok(json!(v["tokenizer_decoded"]==true))));
        // Host warming waits for this CPU component without activating any
        // tensor loader or publishing a numerical request scope.
        shared.insert("load","host-tokenizer",Arc::new(|_|Ok(json!(true))));
        shared.insert("validate_load","host-tokenizer",Arc::new(|v|Ok(json!(v==true))));
        let target=Rc::new(RefCell::new(None::<Target>));let attention=Rc::new(RefCell::new(BTreeMap::new()));
        let installed=Rc::new(RefCell::new(BTreeSet::new()));
        let (profiles,owner,seen)=(target.clone(),resources.clone(),installed.clone());
        let all_names=names.iter().flatten().cloned().collect::<Vec<_>>();
        local.insert("run","resident-weights",Box::new(move |inputs| {
            let targets=profiles.borrow();let target=scope_target(&targets,target_key(inputs)?)?;
            let owner=owner.borrow();let device=&owner.as_ref().ok_or("LTX device missing")?.device;
            let restore=seen.borrow().len()==STAGES.len();
            if restore {
                // The original resident entry restores all missing weights
                // before text. This numerical binding preserves that behavior.
                for (prefix,checkpoint) in &target.checkpoints {
                    let prefix=format!("{prefix}.");
                    let keys=all_names.iter().filter_map(|name|name.strip_prefix(&prefix))
                        .filter(|key|device.cached_weight(&format!("{}\0{key}",checkpoint.path.display())).is_none())
                        .map(str::to_string).collect::<Vec<_>>();
                    if !keys.is_empty() {checkpoint.prefetch_weights(&keys,device).map_err(|e|e.to_string())?;}
                }
            }
            Ok(json!({"request":target.key,"resident_restored":restore,"forward_executed":false}))
        }));
        local.insert("validate_run","resident-weights",Box::new(|v|Ok(json!(v["forward_executed"]==false && v["resident_restored"].is_boolean()))));
        let (profiles,plans)=(target.clone(),attention.clone());
        local.insert("run","text-backend",Box::new(move |inputs| {
            let profiles=profiles.borrow();let profile=scope_target(&profiles,target_key(inputs)?)?;
            if !plans.borrow().contains_key(&profile.text_root) {
                let processor:Value=serde_json::from_slice(&std::fs::read(profile.text_root.join("processor.graph.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
                let count=graph::prepare_attention(&processor).map_err(|e|e.to_string())?;
                plans.borrow_mut().insert(profile.text_root.clone(),count);
            }
            Ok(json!({"request":profile.key,"attention_plans":plans.borrow()[&profile.text_root]}))
        }));
        let (profiles,plans)=(target.clone(),attention.clone());
        local.insert("validate_run","text-backend",Box::new(move |v| {
            let profiles=profiles.borrow();let profile=scope_target(&profiles,target_key(v)?)?;
            Ok(json!(plans.borrow().get(&profile.text_root).is_some_and(|count|v["attention_plans"]==*count)))
        }));
        for (index,name) in STAGES.iter().enumerate() {
            let (profiles,owner,weights,seen,token)=(target.clone(),resources.clone(),loads.clone(),installed.clone(),tokenizer.clone());
            let name=name.to_string();let tensor_names=names[index].clone();let key=format!("install-{name}");
            let prep_name=name.clone();
            local.insert("run",&key,Box::new(move |inputs| {
                let profiles=profiles.borrow();let profile=scope_target(&profiles,target_key(inputs)?)?;
                let owner=owner.borrow();let device=&owner.as_ref().ok_or("LTX device missing")?.device;
                if !seen.borrow().contains(&name) {
                    startup::install(weights.take(&name)?,&profile.checkpoints,device).map_err(|e|e.to_string())?;
                    seen.borrow_mut().insert(name.clone());
                }
                if name=="text" {
                    let text=&profile.checkpoints.iter().find(|(p,_)|*p=="text").ok_or("LTX text checkpoint missing")?.1;
                    if !text.tokenizer_loaded() {
                        text.install_tokenizer(token.lock().unwrap().take().ok_or("LTX tokenizer CPU result missing")?).map_err(|e|e.to_string())?;
                    }
                }
                Ok(json!({"request":profile.key,"component":name,"tensors":tensor_names.len()}))
            }));
            let (profiles,owner)=(target.clone(),resources.clone());let tensor_names=names[index].clone();
            local.insert("validate_run",&key,Box::new(move |v| {
                let profiles=profiles.borrow();let profile=scope_target(&profiles,target_key(v)?)?;
                let owner=owner.borrow();let device=&owner.as_ref().ok_or("LTX device missing")?.device;
                Ok(json!(v["component"]==prep_name && v["tensors"]==tensor_names.len() && ready_tensors(&tensor_names,profile,device)
                    && (prep_name!="text" || profile.checkpoints.iter().any(|(p,c)|*p=="text" && c.tokenizer_loaded()))))
            }));
        }
        let mut preparations=vec![
            prep("model-tokenizer",&["text","host-tokenizer"],&[],&[],true,false,AtUse::Never),
            prep("resident-weights",&STAGES,&[loads.device_component()],&[],false,true,AtUse::Never),
            // The original owner starts processor metadata after device_ready
            // while independent copies continue, before joining text weights.
            prep("text-backend",&["text","connectors"],&[loads.device_component()],&[],false,true,AtUse::Never)];
        for name in STAGES {
            let after=match name {"text"=>vec!["model-tokenizer","resident-weights"],"connectors"=>vec!["install-text","resident-weights"],_=>vec!["resident-weights"]};
            preparations.push(prep(&format!("install-{name}"),&[name],&[name],&after,false,true,AtUse::Always));
        }
        let mut components=loads.components();
        components.push(Component {name:"host-tokenizer".into(),after:vec![],required_preparations:Some(vec!["model-tokenizer".into()]),
            reader_limit_mib:0,prefetch:false,load_at_use:false,initialize:false});
        let runtime=OwnerExecutor::new(Declaration {components,preparations,workers:8,
            reader_budget_mib:options.config.budget_bytes.max(options.retained_budget).div_ceil(1<<20),timeout_seconds:900.,consumers:None},shared,local)?;
        Ok(Self {runtime,resources,target,attention,loads,tokenizer,installed,key:String::new(),generation:0,retained_budget:options.retained_budget})
    }
    pub fn start(&mut self)->Result<Resources> {
        self.runtime.start()?;self.runtime.before_use(None,"native-device")?;
        self.resources.borrow().clone().ok_or_else(||"LTX device creation did not publish resources".into())
    }
    pub fn request(&mut self,text_root:PathBuf,checkpoints:Vec<(&'static str,Rc<Checkpoint>)>)->Result<()> {
        self.generation=self.generation.checked_add(1).ok_or("LTX request generation overflow")?;
        let key=format!("ltx-request-{}",self.generation);
        self.runtime.request(&key,json!({"request":key}))?;
        *self.target.borrow_mut()=Some(Target {key:key.clone(),text_root,checkpoints});self.key=key;Ok(())
    }
    pub fn enter(&mut self,name:&str)->Result<ConsumerGuard> {self.runtime.enter(Some(&self.key),name)}
    pub fn needs_prefetch(&self)->bool {
        let fresh_readers=self.installed.borrow().len()!=STAGES.len() && !self.loads.retained_sources_ready();
        crate::admission::needs_prefetch(true,fresh_readers)
    }
    pub fn check_host_warm(&self)->Result<()> {self.loads.check_host_warm()}
    pub fn warm_host(&mut self,checkpoints:&[(&str,Rc<Checkpoint>)])->Result<bool> {
        self.runtime.before_use(None,"host-tokenizer")?;
        let text=&checkpoints.iter().find(|(prefix,_)|*prefix=="text").ok_or("LTX text checkpoint missing")?.1;
        if !text.tokenizer_loaded() {
            text.install_tokenizer(self.tokenizer.lock().unwrap().take().ok_or("LTX tokenizer CPU result missing")?).map_err(|e|e.to_string())?;
        }
        match self.loads.warm_host(80usize<<30)? {
            Some(hosts)=>{for host in hosts {startup::install_host(host,checkpoints).map_err(|e|e.to_string())?;}Ok(true)},
            None=>Ok(false),
        }
    }
    pub fn report(&self)->Value {
        let mut value=self.runtime.snapshot();
        value["cold_reader_mode"]=json!(if self.retained_budget!=0 {"retained-buffered"}else{"bounded-reusable"});
        value["host_retention_budget_bytes"]=json!(self.retained_budget);value
    }
}
