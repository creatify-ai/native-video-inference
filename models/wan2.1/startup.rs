//! Wan's declarations and numerical bindings. Core owns work submission,
//! context release, stage advancement, waits, validation and cancellation.
use crate::{clip::ClipVisual,cuda::Device,dit::Dit,frontend::Prepared,pipeline,
    t5::T5,vae::Vae,weights::DeviceWeights};
use native_core_infra::{checkpoint,declared_startup::{AtUse,Bindings,Component,Declaration,Preparation,Result},
    model_load::{Loaded,Options},startup_input::RequestInput,
    startup_native::{LoadMode,NativeLoadPlan,WeightStage},startup_owner::{OwnerBindings,OwnerExecutor}};
use serde_json::{json,Value};
use std::{cell::RefCell,rc::Rc,sync::Arc};

#[derive(Default)]
pub(crate) struct Parts<'d> {
    pub device:Option<&'d Device>,pub t5:Option<T5<'d>>,pub clip:Option<ClipVisual<'d>>,
    pub vae:Option<Vae<'d>>,pub dit:Option<Dit<'d>>,pub weights:Vec<Rc<Loaded>>,
    /// With two GPUs: GPU 1's unconditional pass and its DiT copy in flight.
    pub peer:Option<crate::cfg_parallel::CfgWorker>,pub replica:Option<crate::cfg_parallel::Replica>,
}
pub(crate) struct WanStartup {
    // Close/join core before the owner objects and their context can retire.
    runtime:OwnerExecutor<'static>,pub parts:Rc<RefCell<Parts<'static>>>,
    input:Arc<RequestInput<Prepared>>,scope:String,generation:u64,epoch:f64,
}
/// Weight stages in load order. The VAE (0.5 GB) lands first so its encode, the heaviest
/// conditioning step, starts while CLIP and T5 (one stage, so T5's large embedding streams beside
/// the other encoder tensors) and then the DiT stream in.
pub(crate) const VIDEO:&str="vae";
pub(crate) const ENCODERS:&str="encoders";
fn request(inputs:&Value)->Result<&str> {inputs["request"].as_str().ok_or_else(||"Wan startup request key missing".into())}
fn component(name:&str,required:&[&str])->Component {Component {name:name.into(),after:vec![],
    required_preparations:Some(required.iter().map(|s|s.to_string()).collect()),reader_limit_mib:0,prefetch:false,load_at_use:false,initialize:false}}
fn preparation(name:&str,before:&[&str],weights:&[&str],after:&[&str],cpu:bool,at_use:AtUse)->Preparation {
    Preparation {name:name.into(),before:before.iter().map(|s|s.to_string()).collect(),inputs:vec!["request".into()],
        weights:weights.iter().map(|s|s.to_string()).collect(),after:after.iter().map(|s|s.to_string()).collect(),cpu,at_use,initialize:false,serial_group:None}
}
impl WanStartup {
    pub fn new(checkpoint:checkpoint::Declaration,options:Options)->Result<Self> {
        let (mut video,mut encoders,mut dit)=(vec![],vec![],vec![]);
        for request in pipeline::prepared_requests(&checkpoint.manifest) {
            if request.name.starts_with("dit.") {dit.push(request)}
            else if request.name.starts_with("vae.") {video.push(request)}
            else {encoders.push(request)}
        }
        let plan=NativeLoadPlan::new(checkpoint.manifest,checkpoint.roots,options,LoadMode::Staged,vec![
            WeightStage {component:component(VIDEO,&["request-backend","vae-model"]),tensors:video},
            WeightStage {component:component(ENCODERS,&["request-backend","encoder-models"]),tensors:encoders},
            WeightStage {component:component("dit",&["request-backend","dit-model"]),tensors:dit}])?;
        let parts=Rc::new(RefCell::new(Parts::default()));
        let mut bindings=Bindings::default();let mut local=OwnerBindings::default();
        let state=parts.clone();
        // The maintained pipeline already gives the resident process a device
        // with process lifetime. Core closes its uploads before model teardown.
        // Uploads get their own stream: conditioning on the device stream must not
        // queue behind the DiT's copies, and core drains it before each stage is ready.
        let loads=unsafe {plan.bind(&mut bindings,&mut local,move || {
            let device:&'static Device=Box::leak(Box::new(Device::open().map_err(|e|e.to_string())?));
            state.borrow_mut().device=Some(device);Ok(device.upload_stream().map_err(|e|e.to_string())? as usize)
        })}?;
        let input=Arc::new(RequestInput::<Prepared>::default());
        let slot=input.clone();
        bindings.insert("cpu","request-backend",Arc::new(move |inputs| {
            let key=request(inputs)?;slot.prepare(key)?;let value=slot.get(key)?;
            Ok(json!({"request":key,"width":value.width(),"height":value.height(),"frames":value.request.frame_num}))
        }));
        let slot=input.clone();
        bindings.insert("validate_cpu","request-backend",Arc::new(move |receipt| {
            let value=slot.get(request(receipt)?)?;
            Ok(json!(receipt["width"]==value.width() && receipt["height"]==value.height()
                && receipt["frames"]==value.request.frame_num && value.width()>0 && value.height()>0
                && value.request.frame_num%4==1 && !value.image.is_empty()))
        }));
        let slot=input.clone();
        loads.bind_context_work(&mut bindings,"run","request-backend",Arc::new(move |inputs| {
            let key=request(inputs)?;let prepared=slot.get(key)?;
            pipeline::prewarm(&prepared).map_err(|e|e.to_string())?;
            Ok(json!({"request":key,"attention_plans_prepared":true,"forward_executed":false}))
        }))?;
        let slot=input.clone();
        bindings.insert("validate_run","request-backend",Arc::new(move |receipt| {
            slot.get(request(receipt)?)?;
            Ok(json!(receipt["attention_plans_prepared"]==true && receipt["forward_executed"]==false))
        }));
        let (state,weights)=(parts.clone(),loads.clone());
        local.insert("run","vae-model",Box::new(move |_| {
            let mut state=state.borrow_mut();let device=state.device.ok_or("Wan device missing")?;
            if state.vae.is_none() {
                let loaded=Rc::new(weights.take(VIDEO)?);
                let vae=native_core_infra::startup_trace::measure("wan_step","vae load",0,||
                    Vae::load(&DeviceWeights {loaded:loaded.clone(),prefix:"vae.".into()},device)).map_err(|e|e.to_string())?;
                native_core_infra::startup_trace::measure("wan_step","vae sync",0,||device.sync()).map_err(|e|e.to_string())?;
                state.vae=Some(vae);state.weights.push(loaded);
            }
            Ok(json!({"encoders":1}))
        }));
        let state=parts.clone();
        local.insert("validate_run","vae-model",Box::new(move |receipt| {
            let state=state.borrow();Ok(json!(receipt["encoders"]==1 && state.vae.is_some()))
        }));
        let (state,weights)=(parts.clone(),loads.clone());
        local.insert("run","encoder-models",Box::new(move |_| {
            let mut state=state.borrow_mut();let device=state.device.ok_or("Wan device missing")?;
            if state.t5.is_none() {
                let loaded=Rc::new(weights.take(ENCODERS)?);
                let part=|prefix:&str|DeviceWeights {loaded:loaded.clone(),prefix:prefix.into()};
                let clip=native_core_infra::startup_trace::measure("wan_step","clip load",0,||ClipVisual::load(&part("clip."),device)).map_err(|e|e.to_string())?;
                let t5=native_core_infra::startup_trace::measure("wan_step","t5 load",0,||T5::load(&part("t5."),device)).map_err(|e|e.to_string())?;
                native_core_infra::startup_trace::measure("wan_step","encoders sync",0,||device.sync()).map_err(|e|e.to_string())?;
                state.clip=Some(clip);state.t5=Some(t5);state.weights.push(loaded);
            }
            Ok(json!({"encoders":2}))
        }));
        let state=parts.clone();
        local.insert("validate_run","encoder-models",Box::new(move |receipt| {
            let state=state.borrow();Ok(json!(receipt["encoders"]==2 && state.clip.is_some() && state.t5.is_some()))
        }));
        let (state,weights,slot)=(parts.clone(),loads.clone(),input.clone());
        local.insert("run","dit-model",Box::new(move |inputs| {
            let prepared=slot.get(request(inputs)?)?;let mut state=state.borrow_mut();let state=&mut *state;
            let device=state.device.ok_or("Wan device missing")?;
            if state.dit.is_none() {
                let loaded=Rc::new(weights.take("dit")?);
                // GPU 1 copies these weights over the peer link while GPU 0 builds its DiT.
                if let Some(peer)=&state.peer {
                    state.replica=Some(peer.begin_replicate(&loaded,pipeline::grid(&prepared)).map_err(|e|e.to_string())?);
                }
                state.dit=Some(Dit::load_from(&DeviceWeights {loaded:loaded.clone(),prefix:"dit.".into()},
                    pipeline::grid(&prepared),device).map_err(|e|e.to_string())?);
                state.weights.push(loaded);
            } else {state.dit.as_mut().unwrap().set_grid(pipeline::grid(&prepared)).map_err(|e|e.to_string())?;}
            device.sync().map_err(|e|e.to_string())?;
            Ok(json!({"request":request(inputs)?,"grid":[prepared.latent_frames(),prepared.lat_h,prepared.lat_w]}))
        }));
        let (state,slot)=(parts.clone(),input.clone());
        local.insert("validate_run","dit-model",Box::new(move |receipt| {
            let prepared=slot.get(request(receipt)?)?;let state=state.borrow();
            Ok(json!(state.dit.as_ref().is_some_and(|dit|dit.grid==pipeline::grid(&prepared))
                && receipt["grid"]==json!([prepared.latent_frames(),prepared.lat_h,prepared.lat_w])))
        }));
        let declaration=Declaration {components:loads.components(),preparations:vec![
            preparation("request-backend",&[VIDEO,ENCODERS,"dit"],&[loads.device_component()],&[],true,AtUse::Never),
            preparation("vae-model",&[VIDEO],&[VIDEO],&["request-backend"],false,AtUse::Never),
            preparation("encoder-models",&[ENCODERS],&[ENCODERS],&["request-backend"],false,AtUse::Never),
            // Preserve activation/VRAM lifetime: conditioning ends before the
            // DiT's request-sized scratch is constructed on the owner lane.
            preparation("dit-model",&["dit"],&["dit"],&["request-backend"],false,AtUse::Always)],
            workers:8,reader_budget_mib:options.config.budget_bytes.div_ceil(1<<20).max(options.ahead_budget.div_ceil(1<<20)),
            timeout_seconds:900.,consumers:None};
        let runtime=OwnerExecutor::new(declaration,bindings,local)?;
        let epoch=pipeline::unix_now()-runtime.elapsed_seconds();
        Ok(Self {runtime,parts,input,scope:String::new(),generation:0,epoch})
    }
    pub fn start(&self)->Result<()> {self.runtime.start()}
    pub fn request(&mut self,front:impl FnOnce()->Result<Prepared>+Send+'static)->Result<()> {
        self.generation=self.generation.checked_add(1).ok_or("Wan request generation overflow")?;
        self.scope=format!("wan-request-{}",self.generation);
        self.input.submit(&self.scope,front)?;
        self.runtime.request(&self.scope,json!({"request":self.scope}))?;Ok(())
    }
    pub fn ready(&mut self,name:&str)->Result<()> {self.runtime.before_use(Some(&self.scope),name)}
    pub fn invoke<T>(&mut self,name:&str,work:impl FnOnce()->crate::Result<T>)->crate::Result<T> {
        Ok(self.runtime.invoke(Some(&self.scope),name,||work().map_err(|e|e.to_string()))?)
    }
    pub fn prepared(&self)->Result<Arc<Prepared>> {self.input.get(&self.scope)}
    pub fn take_prepared(&self)->Result<Prepared> {self.input.take(&self.scope)}
    pub fn report(&self)->Value {self.runtime.snapshot()}
    pub fn marks(&self)->Vec<(String,f64)> {
        let state=self.runtime.snapshot();
        state["events"].as_array().into_iter().flatten().filter_map(|event| {
            let pair=(event["component"].as_str()?,event["phase"].as_str()?);
            let name=match pair {
                ("native-readers","ready")=>"readers_started",("native-device","ready")=>"cuda_context",
                ("vae","ready")=>"vae_weights_on_device",("encoders","ready")=>"encoder_weights_on_device",
                ("request-backend","cpu_prepare_ready")=>"frontend",("request-backend","prepare_ready")=>"plans_prewarmed",
                ("vae-model","prepare_ready")=>"vae_ready",("encoder-models","prepare_ready")=>"encoders_ready",
                ("dit","ready")=>"dit_weights_on_device",("dit-model","prepare_ready")=>"dit_ready",_=>return None};
            Some((name.into(),self.epoch+event["elapsed_seconds"].as_f64()?))
        }).collect()
    }
}
