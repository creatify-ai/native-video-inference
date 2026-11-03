//! Native weight/context bindings for `DeclaredStartup`, using its one policy.
//! A model declares tensor groups, preparations and its device constructor.
//! Core starts readers, publishes the context, advances uploads and joins them.
//! Non-Send numerical objects still stay on the `OwnerExecutor` lane.
use crate::checkpoint::Manifest;
use crate::declared_startup::{Bindings, Component, Result, Work};
use crate::model_load::{self, Components, HostWeights, IndependentUploads, Loaded, Options, Request, RestHandle, Staged};
use crate::startup_owner::OwnerBindings;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}};

const READERS: &str = "native-readers";
const DEVICE: &str = "native-device";

#[derive(Clone, Copy)]
pub enum LoadMode { Staged, Independent, IndependentAtUse }

pub struct WeightStage {
    pub component: Component,
    pub tensors: Vec<Request>,
}

pub struct NativeLoadPlan {
    manifest: Manifest, roots: HashMap<String, PathBuf>, options: Options,
    mode: LoadMode, stages: Vec<WeightStage>,storage_layout:Option<(crate::checkpoint::Declaration,String)>,file_read_policy:crate::weight_loader::FileReadPolicy,read_schedule:crate::weight_loader::ReadSchedule,component_readers:Option<Vec<crate::weight_loader::Config>>,
}

enum Pending { Staged(Staged), Independent(Components) }
impl Pending {
    unsafe fn activate(&mut self) -> Result<()> {
        if let Self::Independent(pending) = self { unsafe { pending.device_ready()?; } }
        Ok(())
    }
    unsafe fn first(&mut self, stream: usize) -> Result<Loaded> {
        match self {
            Self::Staged(p) => unsafe { p.finish_first(stream as *mut _) },
            Self::Independent(p) => unsafe { p.finish_first(stream as *mut _) },
        }
    }
    fn rest(self) -> Result<RestHandle> {
        match self { Self::Staged(p) => p.spawn_rest(), Self::Independent(p) => p.spawn_rest() }
    }
}
struct Progress {
    pending: Option<Pending>, rest: Option<RestHandle>,
    independent:Option<IndependentUploads>,retained: Option<Vec<Arc<HostWeights>>>,
}
struct Context { identity: usize, stream: usize }
struct Shared {
    progress: Mutex<Progress>, context: Mutex<Option<Context>>,
    // A waiting later upload must not lock the first stage's typed result.
    values: Mutex<BTreeMap<String, Loaded>>, closed: AtomicBool,retained_ready:AtomicBool,completed:AtomicUsize,
}

#[derive(Clone)]
pub struct NativeLoads { shared: Arc<Shared>, components: Vec<Component>, manifest:Arc<Manifest> }
impl NativeLoads {
    pub fn components(&self) -> Vec<Component> { self.components.clone() }
    /// Share the exact admitted checkpoint metadata already held by readers.
    /// Model adapters must not deep-copy every tensor/guard for each component.
    /// Holding this Arc neither starts I/O nor publishes weight readiness.
    pub fn manifest(&self)->Arc<Manifest> {self.manifest.clone()}

    pub fn readers_component(&self)->&'static str {READERS}
    pub fn device_component(&self) -> &'static str { DEVICE }
    // This is an admission fact, not a request to join a GPU upload. The
    // progress owner can hold its lock for an entire checkpoint stage.
    pub fn retained_sources_ready(&self)->bool {
        !self.shared.closed.load(Ordering::Acquire) && self.shared.retained_ready.load(Ordering::Acquire)
    }
    /// CPU-only warm may adopt readers before upload, or reuse a completed
    /// host cache. An in-flight GPU cohort is never consumed by this operation.
    pub fn check_host_warm(&self)->Result<()> {
        if self.shared.closed.load(Ordering::Acquire) {return Err("native startup cancelled".into());}
        let progress=self.shared.progress.try_lock().map_err(|_|"host warmup cannot take over an in-flight device upload")?;
        if progress.retained.is_some() || self.shared.completed.load(Ordering::Acquire)==self.components.len()-2
            || matches!(&progress.pending,Some(Pending::Independent(p)) if !p.device_started()) {Ok(())}
        else {Err("host warmup must precede device upload; pending uploads remain available to infer".into())}
    }
    /// Take over the admitted CPU readers without reopening their sources.
    /// `None` means GPU loading already completed without retained sources.
    /// Call on the live device owner; registration belongs to that context.
    pub fn warm_host(&self,budget:usize)->Result<Option<Vec<Arc<HostWeights>>>> {
        self.check_host_warm()?;
        let mut progress=self.shared.progress.try_lock().map_err(|_|"host takeover raced with device upload")?;
        if let Some(hosts)=&progress.retained {return Ok(Some(hosts.clone()));}
        if self.shared.completed.load(Ordering::Acquire)==self.components.len()-2 {return Ok(None);}
        if !matches!(&progress.pending,Some(Pending::Independent(p)) if !p.device_started()) {
            return Err("host takeover raced with device upload".into());
        }
        let identity=self.shared.context.lock().unwrap().as_ref().ok_or("native context not ready")?.identity;
        let _context=unsafe{crate::device_context::CurrentContext::enter(identity)}.map_err(|e|e.to_string())?;
        let Some(Pending::Independent(pending))=progress.pending.take() else {unreachable!()};
        let hosts=pending.finish_host(budget)?;
        progress.retained=Some(hosts.clone());self.shared.retained_ready.store(true,Ordering::Release);Ok(Some(hosts))
    }
    /// Called by a model's owner-thread constructor after its declared weight
    /// dependency has passed. Ownership moves once; readiness stays in core.
    pub fn take(&self, name: &str) -> Result<Loaded> {
        self.shared.values.lock().unwrap().remove(name)
            .ok_or_else(||format!("native startup weights unavailable or already consumed: {name}"))
    }
    /// Bind a request/backend numerical function to the admitted context. Its
    /// Preparation must name `device_component()` in `weights`.
    pub fn bind_context_work(&self, bindings: &mut Bindings, operation: &str, name: &str, work: Work) -> Result<()> {
        if bindings.has(operation, name) { return Err("duplicate native context callback".into()); }
        let shared = self.shared.clone();
        bindings.insert(operation, name, Arc::new(move |inputs| {
            if shared.closed.load(Ordering::Acquire) { return Err("native startup cancelled".into()); }
            let identity = shared.context.lock().unwrap().as_ref()
                .ok_or("native context dependency is not ready")?.identity;
            let _context = unsafe { crate::device_context::CurrentContext::enter(identity) }
                .map_err(|e|e.to_string())?;
            work(inputs)
        }));
        Ok(())
    }
}

impl NativeLoadPlan {
    /// Metadata-only admission. This does not create a thread, context or reader.
    pub fn new(manifest: Manifest, roots: HashMap<String,PathBuf>, options: Options,
               mode: LoadMode, stages: Vec<WeightStage>) -> Result<Self> {
        if stages.is_empty() { return Err("native startup needs tensor components".into()); }
        let mut names = BTreeSet::from([READERS.to_string(), DEVICE.to_string()]);
        let mut tensors = BTreeSet::new();
        for stage in &stages {
            if stage.component.name.is_empty() || !names.insert(stage.component.name.clone())
                || stage.tensors.is_empty() || !stage.component.after.is_empty()
                || stage.component.prefetch || stage.component.reader_limit_mib != 0
                || (stage.component.load_at_use && matches!(mode,LoadMode::Staged)) {
                return Err("invalid native tensor component; core owns its loader edges and shared readers".into());
            }
            for request in &stage.tensors {
                manifest.tensor(&request.name)?;
                if matches!(mode,LoadMode::IndependentAtUse) && manifest.tensor(&request.name)?.dtype!=request.dtype {
                    return Err("host-warmable components require matching stored/runtime dtypes".into());
                }
                if !tensors.insert(request.name.clone()) { return Err("native tensor appears in multiple components".into()); }
            }
        }
        options.config.validate()?;
        if !matches!(mode,LoadMode::Staged) && options.ahead_budget!=0 {
            return Err("independent components share one reader budget, without a staged read-ahead budget".into());
        }
        if !matches!(mode,LoadMode::Staged) && options.retained_budget!=0 {
            model_load::component_retention_bounds(&manifest,&stages.iter().map(|s|s.tensors.clone()).collect::<Vec<_>>(),options)?;
        }
        let plan=Self {manifest, roots, options, mode, stages,file_read_policy:crate::weight_loader::FileReadPolicy::Auto,
            read_schedule:crate::weight_loader::ReadSchedule::ByteFair,component_readers:None,storage_layout:None};
        let plan=match std::env::var("NATIVE_LOAD_COMPONENT_READERS"){
            Ok(raw)=>plan.with_component_readers(serde_json::from_str(&raw).map_err(|e|format!("invalid component reader quotas: {e}"))?),
            Err(std::env::VarError::NotPresent)=>Ok(plan),Err(_)=>Err("invalid component reader quotas".into()),
        }?;
        match (std::env::var_os("NATIVE_WEIGHT_LAYOUT_MANIFEST"),std::env::var("NATIVE_WEIGHT_LAYOUT_SHA256"),std::env::var("NATIVE_WEIGHT_LAYOUT_ROOTS")) {
            (None,Err(std::env::VarError::NotPresent),Err(std::env::VarError::NotPresent))=>Ok(plan),
            (Some(path),Ok(sha256),Ok(roots))=>plan.with_storage_layout(crate::checkpoint::Source{
                manifest:path.into(),sha256,roots:serde_json::from_str(&roots).map_err(|e|format!("invalid storage layout roots: {e}"))?}),
            _=>Err("storage layout requires its manifest, SHA256 and roots together".into()),
        }
    }

    /// Substitute only storage locations for the same complete checkpoint.
    /// The original admitted metadata remains available to model adapters.
    pub fn with_storage_layout(mut self,source:crate::checkpoint::Source)->Result<Self>{
        let physical=source.load()?;self.manifest.validate_equivalent_tensors(&physical.manifest)?;
        self.storage_layout=Some((physical,source.sha256));Ok(self)
    }

    /// Configure independent bounded readers without changing any tensor,
    /// integrity, buffer-budget or CUDA upload policy. Legacy plans keep Auto.
    pub fn with_file_read_policy(mut self,policy:crate::weight_loader::FileReadPolicy)->Result<Self> {
        if matches!(self.mode,LoadMode::Staged){return Err("file-read policy requires independent component loads".into());}
        self.file_read_policy=policy;
        if let Some(configs)=&self.component_readers{
            self.component_readers=Some(crate::weight_loader::component_reader_configs(self.options.config,
                &configs.iter().map(|c|c.workers).collect::<Vec<_>>(),self.options.direct,policy)?);
        }Ok(self)
    }

    /// Give each independent component its own reader cohort. All quotas
    /// share one explicit aggregate envelope; they never multiply its budget.
    pub fn with_component_readers(mut self,quotas:BTreeMap<String,usize>)->Result<Self>{
        if matches!(self.mode,LoadMode::Staged)||self.options.retained_budget!=0{
            return Err("component reader quotas require bounded independent streaming loads".into());
        }
        if quotas.keys().collect::<BTreeSet<_>>()!=self.stages.iter().map(|s|&s.component.name).collect::<BTreeSet<_>>(){
            return Err("reader quotas must name every admitted weight component exactly once".into());
        }
        let workers=self.stages.iter().map(|s|quotas[&s.component.name]).collect::<Vec<_>>();
        self.component_readers=Some(crate::weight_loader::component_reader_configs(self.options.config,&workers,self.options.direct,self.file_read_policy)?);
        Ok(self)
    }

    /// Select allocation within the same reader/buffer budget. This changes
    /// no component/preparation edge, tensor layout, verification or upload.
    pub fn with_read_schedule(mut self,schedule:crate::weight_loader::ReadSchedule)->Result<Self>{
        if matches!(self.mode,LoadMode::Staged){return Err("read schedule requires independent component loads".into());}
        self.read_schedule=schedule;Ok(self)
    }

    /// Install transport bindings, leaving all scheduling to DeclaredStartup.
    /// `open_device` runs on the owner lane and returns its live CUDA stream.
    ///
    /// # Safety
    /// The device/context created by `open_device` must outlive the executor and
    /// returned allocations. Close/drop the executor before destroying that
    /// context, including on errors. Core joins every background upload first.
    pub unsafe fn bind<'a>(self, bindings: &mut Bindings, local: &mut OwnerBindings<'a>,
        mut open_device: impl FnMut() -> Result<usize> + 'a) -> Result<NativeLoads> {
        if bindings.has("cancel", "") { return Err("native load lifetime already bound".into()); }
        let deferred=self.stages.iter().map(|s|s.component.load_at_use).collect::<Vec<_>>();
        if deferred.iter().any(|v|*v) && self.component_readers.is_none(){
            return Err("deferred native components require explicitly budgeted per-component readers".into());
        }
        let shared = Arc::new(Shared {progress: Mutex::new(Progress {pending:None,rest:None,independent:None,retained:None}),
            context:Mutex::new(None), values:Mutex::new(BTreeMap::new()), closed:AtomicBool::new(false),
            retained_ready:AtomicBool::new(false),completed:AtomicUsize::new(0)});
        let reader_bytes=self.options.config.budget_bytes.max(self.options.ahead_budget).max(self.options.retained_budget);
        let mut components = vec![Component {name:READERS.into(), after:vec![], required_preparations:Some(vec![]),
            reader_limit_mib:reader_bytes.div_ceil(1<<20), prefetch:false,load_at_use:false,initialize:false},
            Component {name:DEVICE.into(), after:vec![], required_preparations:Some(vec![]),
                reader_limit_mib:0,prefetch:false,load_at_use:false,initialize:false}];
        for stage in &self.stages {
            let mut component = stage.component.clone();
            component.after=if matches!(self.mode,LoadMode::Staged) && components.len()>2 {
                vec![components.last().unwrap().name.clone()]
            }else{vec![READERS.into(),DEVICE.into()]};
            component.load_at_use|=matches!(self.mode,LoadMode::IndependentAtUse);
            components.push(component);
        }
        let stage_names = self.stages.iter().map(|s|s.component.name.clone()).collect::<Vec<_>>();
        let expected = self.stages.iter().map(|s|s.tensors.iter().map(|t|t.name.clone()).collect::<BTreeSet<_>>()).collect::<Vec<_>>();
        let requests = Arc::new(self.stages.into_iter().map(|s|s.tensors).collect::<Vec<_>>());
        let logical_manifest=Arc::new(self.manifest);
        let (manifest,roots,storage_sha)=match self.storage_layout {
            Some((declaration,pin))=>(Arc::new(declaration.manifest),declaration.roots,Some(pin)),
            None=>(logical_manifest.clone(),self.roots,None),
        };
        let schedule=crate::weight_loader::ReadSchedule::from_env(self.read_schedule)?;
        let source = Mutex::new(Some((manifest.clone(),roots,requests.clone(),self.options,self.mode,self.file_read_policy,schedule,self.component_readers)));
        let reader_names=stage_names.clone();
        let reader_deferred=deferred.clone();
        let state = shared.clone();
        bindings.insert("load", READERS, Arc::new(move |_| {
            let (manifest,roots,requests,options,mode,policy,schedule,configs)=source.lock().unwrap().take().ok_or("native readers already started")?;
            let per_component=configs.is_some();
            let component_workers=configs.as_ref().map(|configs|reader_names.iter().cloned().zip(configs.iter().zip(requests.iter())
                .map(|(c,requests)|c.workers.min(requests.len()))).collect::<BTreeMap<_,_>>()).unwrap_or_default();
            let workers=if per_component{component_workers.values().sum()}else{options.config.workers.min(requests.iter().map(Vec::len).sum())};
            if state.closed.load(Ordering::Acquire) {return Err("native startup cancelled".into());}
            let pending = match mode {
                LoadMode::Staged => Pending::Staged(model_load::start_staged((*manifest).clone(),roots,(*requests).clone(),options)?),
                LoadMode::Independent | LoadMode::IndependentAtUse => Pending::Independent(if let Some(configs)=configs{
                    model_load::start_components_with_deferred_readers(&manifest,&roots,(*requests).clone(),options,policy,schedule,configs,&reader_deferred)?
                }else{model_load::start_components_scheduled(&manifest,&roots,(*requests).clone(),options,policy,schedule)?}),
            };
            let mut progress=state.progress.lock().unwrap();
            if state.closed.load(Ordering::Acquire) {return Err("native startup cancelled".into());}
            progress.pending=Some(pending);
            let receipt=json!({"event":"native-reader-policy","readers_started":true,"shared_budget_bytes":reader_bytes,
                "file_read_policy":policy.effective()?.as_str(),"direct_upload":options.direct,
                "upload_depth":crate::weight_loader::configured_upload_slot_count(options.config,options.direct)?,"read_schedule":schedule.as_str(),"layout":if per_component{"per_component"}else{"shared"},
                "storage_manifest_sha256":storage_sha,"logical_tensors_unchanged":true,
                "workers":workers,"component_workers":component_workers,"chunk_bytes":options.config.chunk_bytes,"pinned_slots":options.config.pinned_slots,
                "deferred_components":reader_names.iter().zip(&reader_deferred).filter_map(|(name,defer)|defer.then_some(name)).collect::<Vec<_>>(),
                "initial_readers_started":reader_names.iter().zip(&reader_deferred).filter_map(|(name,defer)|(!defer).then_some(name)).collect::<Vec<_>>(),
                "initial_reader_workers":if per_component{reader_names.iter().zip(&reader_deferred).filter(|(_,defer)|!**defer).map(|(name,_)|component_workers[name]).sum::<usize>()}else{workers}});
            eprintln!("{receipt}");Ok(receipt)
        }));
        bindings.insert("validate_load", READERS, Arc::new(move |v|Ok(json!(v["readers_started"]==true && v["shared_budget_bytes"]==reader_bytes))));
        let state=shared.clone();
        local.insert("load", DEVICE, Box::new(move |_| {
            if state.closed.load(Ordering::Acquire) {return Err("native startup cancelled".into());}
            let stream=open_device()?;
            let identity=crate::device_context::current_identity().map_err(|e|e.to_string())?;
            *state.context.lock().unwrap()=Some(Context {identity,stream});
            Ok(json!({"context":identity,"stream":stream}))
        }));
        bindings.insert("validate_load", DEVICE, Arc::new(|v|Ok(json!(v["context"].as_u64().is_some_and(|id|id!=0) && v["stream"].as_u64().is_some()))));
        let stage_count=stage_names.len();
        for (index,name) in stage_names.iter().enumerate() {
            let (state,key,manifest,requests)=(shared.clone(),name.clone(),manifest.clone(),requests.clone());
            let deferred=deferred.clone();
            if !matches!(self.mode,LoadMode::Staged) {
                bindings.insert_async("load",name,Arc::new(move |_,done| {
                    let receiver=(|| {
                        if state.closed.load(Ordering::Acquire) {return Err("native startup cancelled".into());}
                        let identity=state.context.lock().unwrap().as_ref().ok_or("native context not ready")?.identity;
                        let _context=unsafe{crate::device_context::CurrentContext::enter(identity)}.map_err(|e|e.to_string())?;
                        let mut progress=state.progress.lock().unwrap();
                        if progress.independent.is_none() {
                            let pending=match progress.pending.take() {
                                Some(Pending::Independent(p))=>p,
                                Some(Pending::Staged(_))=>return Err("independent binding received staged readers".into()),
                                None=>unsafe{Components::from_host_with_deferred(manifest.clone(),(*requests).clone(),
                                    progress.retained.as_ref().ok_or("native independent sources unavailable")?.clone(),&deferred)}?,
                            };
                            let mut pending=pending;unsafe{pending.device_ready()?;}
                            progress.independent=Some(pending.into_uploads()?);
                        }
                        progress.independent.as_mut().unwrap().take(index)
                    })();
                    match receiver {
                        Err(error)=>done.send(Err(error)),
                        Ok(receiver)=>{
                            let (state,key)=(state.clone(),key.clone());
                            receiver.subscribe(move |result| {
                                let context=state.context.lock().unwrap().as_ref().map(|c|c.identity);
                                let _context=context.map(|id|unsafe{crate::device_context::CurrentContext::enter(id)})
                                    .transpose().map_err(|e|e.to_string());
                                let _context=match _context {Ok(context)=>context,Err(error)=>{done.send(Err(error));return;}};
                                let result=result.and_then(|(loaded,_)| {
                                    let receipt=json!({"component":key,"stats":loaded.stats});
                                    let mut values=state.values.lock().unwrap();
                                    if state.closed.load(Ordering::Acquire) {return Err("native startup cancelled".into());}
                                    values.insert(key.clone(),loaded);drop(values);
                                    state.completed.fetch_add(1,Ordering::Release);Ok(receipt)
                                });
                                done.send(result);
                            });
                        },
                    }
                }));
            } else {
            let load=move |_: &Value| {
                let mut progress=state.progress.lock().unwrap();
                if state.closed.load(Ordering::Acquire) {return Err("native startup cancelled".into());}
                let identity=state.context.lock().unwrap().as_ref().ok_or("native context not ready")?.identity;
                let _context=unsafe {crate::device_context::CurrentContext::enter(identity)}.map_err(|e|e.to_string())?;
                let loaded=if index==0 {
                    let mut pending=match progress.pending.take() {
                        Some(pending)=>pending,
                        None=>Pending::Independent(unsafe{Components::from_host(manifest.clone(),(*requests).clone(),
                            progress.retained.as_ref().ok_or("native first stage unavailable")?.clone())}?),
                    };
                    let stream=state.context.lock().unwrap().as_ref().ok_or("native context not ready")?.stream;
                    let loaded=unsafe {pending.first(stream)?};
                    progress.rest=Some(pending.rest()?);
                    if stage_count==1 {progress.rest.take().unwrap().join().map_err(|_|"native upload thread panicked")??;}
                    loaded
                } else {
                    let (loaded,_)=progress.rest.as_mut().ok_or("native later stage unavailable")?.next_stage()?;
                    if index+1==stage_count {
                        let extra=progress.rest.take().unwrap().join().map_err(|_|"native upload thread panicked")??;
                        if !extra.is_empty() {return Err("native upload returned undeclared stages".into());}
                    }
                    loaded
                };
                let receipt=json!({"component":key,"stats":loaded.stats});
                state.values.lock().unwrap().insert(key.clone(),loaded);
                state.completed.fetch_add(1,Ordering::Release);
                Ok(receipt)
            };
            // Uploads are context-bound core mechanisms, not thread-affine
            // numerical objects. Keep the owner free to prepare metadata while
            // the first stage drains, as in the original independent loader.
            bindings.insert("load",name,Arc::new(load));
            }
            let (key,tensors,state)=(name.clone(),expected[index].clone(),shared.clone());
            bindings.insert("validate_load",name,Arc::new(move |v| {
                let values=state.values.lock().unwrap();
                let valid=values.get(&key).is_some_and(|loaded|loaded.tensors.len()==tensors.len()
                    && tensors.iter().all(|name|loaded.tensors.contains_key(name)));
                Ok(json!(v["component"]==key && valid && v["stats"]["tensors"]==tensors.len()))
            }));
        }
        let state=shared.clone();
        bindings.insert("cancel","",Arc::new(move |_| {
            state.closed.store(true,Ordering::Release);
            state.retained_ready.store(false,Ordering::Release);
            let identity=state.context.lock().unwrap().as_ref().map(|c|c.identity);
            let _context=identity.map(|id|unsafe {crate::device_context::CurrentContext::enter(id)}).transpose().map_err(|e|e.to_string())?;
            let mut progress=state.progress.lock().unwrap();
            progress.pending.take();
            progress.independent.take();
            let joined=progress.rest.take().map(|rest|rest.join().map_err(|_|"native upload thread panicked during close".to_string()).and_then(|r|r));
            state.values.lock().unwrap().clear();
            progress.retained.take();
            if let Some(result)=joined {result?;}
            Ok(Value::Null)
        }));
        Ok(NativeLoads {shared,components,manifest:logical_manifest})
    }
}

#[cfg(test)]
mod native_startup_control_tests {
    use super::*;
    #[test]
    fn independent_declarations_have_no_receipt_order_edges_and_admit_without_io() {
        let hash="0".repeat(64);
        let manifest=Manifest::parse(&serde_json::to_vec(&json!({"schema":"native-checkpoint-v1","segment_bytes":1024,
            "files":{"weights":{"root":"model","path":"absent.raw","bytes":4,"sha256":hash}},
            "tensors":{
                "a":{"file":"weights","offset":0,"bytes":2,"dtype":"BF16","shape":[1],"sha256":hash,"head_sha256":hash,"tail_sha256":hash},
                "b":{"file":"weights","offset":2,"bytes":2,"dtype":"BF16","shape":[1],"sha256":hash,"head_sha256":hash,"tail_sha256":hash}}})).unwrap(),None).unwrap();
        for mode in [LoadMode::Staged,LoadMode::Independent,LoadMode::IndependentAtUse] {
            let options=Options {config:crate::weight_loader::Config{workers:1,chunk_bytes:4096,pinned_slots:1,budget_bytes:16384},
                direct:false,balanced:true,verify:model_load::Verify::Full,staging_slots:1,ahead_budget:0,retained_budget:0};
            let stages=["a","b"].into_iter().map(|name|WeightStage {
                component:Component{name:name.into(),after:vec![],required_preparations:Some(vec![]),reader_limit_mib:0,prefetch:false,load_at_use:false,initialize:false},
                tensors:vec![Request{name:name.into(),dtype:crate::checkpoint::Dtype::BF16}]}).collect();
            let plan=NativeLoadPlan::new(manifest.clone(),HashMap::new(),options,mode,stages).unwrap();
            let mut bindings=Bindings::default();let mut local=OwnerBindings::default();
            let loads=unsafe{plan.bind(&mut bindings,&mut local,||panic!("admission opened CUDA"))}.unwrap();
            let components=loads.components();
            assert!(components[0].after.is_empty());assert!(components[1].after.is_empty());
            assert_eq!(components[2].after,vec![READERS,DEVICE]);
            assert_eq!(components[3].after,if matches!(mode,LoadMode::Staged){vec!["a"]}else{vec![READERS,DEVICE]});
            for component in &components[2..] {assert_eq!(component.load_at_use,matches!(mode,LoadMode::IndependentAtUse));}
            // OwnerExecutor installs the one remaining device binding before
            // ordinary full-plan admission; no reader/file/context is started.
            let runtime=crate::startup_owner::OwnerExecutor::new(crate::declared_startup::Declaration {
                components,preparations:vec![],workers:1,reader_budget_mib:1,timeout_seconds:1.,consumers:None},bindings,local).unwrap();
            drop(runtime);
        }
    }
    #[test]
    fn metadata_queries_never_join_a_running_upload() {
        let shared=Arc::new(Shared {progress:Mutex::new(Progress {pending:None,rest:None,independent:None,retained:None}),
            context:Mutex::new(None),values:Mutex::new(BTreeMap::new()),closed:AtomicBool::new(false),
            retained_ready:AtomicBool::new(false),completed:AtomicUsize::new(0)});
        let hash="0".repeat(64);
        let manifest=Manifest::parse(&serde_json::to_vec(&json!({"schema":"native-checkpoint-v1","segment_bytes":1024,
            "files":{"weights":{"root":"model","path":"absent.raw","bytes":2,"sha256":hash}},
            "tensors":{"a":{"file":"weights","offset":0,"bytes":2,"dtype":"BF16","shape":[1],
                "sha256":hash,"head_sha256":hash,"tail_sha256":hash}}})).unwrap(),None).unwrap();
        let loads=NativeLoads {shared:shared.clone(),components:vec![],manifest:Arc::new(manifest)};
        let upload=shared.progress.lock().unwrap();
        let worker=loads.clone();let (send,receive)=std::sync::mpsc::channel();
        let thread=std::thread::spawn(move || {
            send.send((worker.retained_sources_ready(),worker.check_host_warm().is_err())).unwrap();
        });
        let reply=receive.recv_timeout(std::time::Duration::from_millis(500));
        drop(upload);thread.join().unwrap();
        assert_eq!(reply.unwrap(),(false,true));
        shared.retained_ready.store(true,Ordering::Release);assert!(loads.retained_sources_ready());
        shared.closed.store(true,Ordering::Release);assert!(!loads.retained_sources_ready());
    }
}
