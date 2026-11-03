//! Standard resident-model actor. A model supplies layout and inference;
//! this layer owns admission, tickets, cancellation, results and ABI memory.
use crate::{backend_abi::{self,Bytes},native_backend::Pin,Result};
use serde::Deserialize;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::BTreeMap,path::{Path,PathBuf},sync::{atomic::{AtomicBool,Ordering},mpsc,Arc,Mutex},thread};

pub type Reply=std::result::Result<Value,String>;
#[derive(Clone)]
pub struct Context {state:Arc<Mutex<State>>,cancelled:Arc<AtomicBool>,decode_gate:Option<crate::phase_gate::Pin>}
impl Context {
    pub fn phase(&self,phase:&str){if let Ok(mut state)=self.state.lock(){state.phase=phase.into();}}
    pub fn checkpoint(&self)->Result<()>{
        if self.cancelled.load(Ordering::Acquire){return Err("native request cancelled".into());}Ok(())
    }
    /// Models declare a terminal phase before allocating its GPU workspace.
    /// The framework submits lookahead and resumes this phase beside DMA.
    pub fn before_gpu_phase(&self, stage:&str, geometry:&[usize])->Result<Option<Value>> {
        self.checkpoint()?;self.phase(stage);
        let receipt=self.decode_gate.as_ref().map(|pin|crate::phase_gate::acknowledge(pin,stage,geometry)).transpose()?;
        self.checkpoint()?;Ok(receipt)
    }
    /// The owning actor drops only weights whose last use has completed.
    /// Publish the retired phase before the framework can start foreign DMA.
    pub fn before_gpu_phase_with_retirement(&self,stage:&str,geometry:&[usize],retire:impl FnOnce()->Result<Value>)->Result<Option<Value>>{
        self.checkpoint()?;self.phase(stage);
        let receipt=self.decode_gate.as_ref().map(|pin|crate::phase_gate::acknowledge_with_retirement(pin,stage,geometry,||{
            let value=retire()?;self.phase(crate::phase_gate::DECODE_WEIGHTS_RETIRED);Ok(value)
        })).transpose()?;
        self.checkpoint()?;Ok(receipt)
    }
}
pub trait Session {
    /// Observational current state. After warm and framework pin completion,
    /// this snapshot overrides colliding metadata/resource fields in the warm
    /// receipt. Target-only warm information must use distinct operation keys.
    fn metadata(&self)->Result<Value>;
    fn warm(&mut self,profile:&Path)->Result<Value>;
    fn prefetch(&mut self,profile:&Path,budget:usize)->Result<Value>;
    fn prefetch_components(&mut self,profile:&Path,budget:usize,components:Option<&[String]>)->Result<Value>{
        if components.is_some(){return Err("native backend does not admit component preload".into());}
        self.prefetch(profile,budget)
    }
    fn infer(&mut self,profile:&Path,request:&Path,output:&Path,context:&Context)->Result<Value>;
    fn evict(&mut self)->Result<Value>;
    fn audit(&mut self)->Result<Value>{Err("native weight audit is not admitted".into())}
    /// Validation is output observation, never a serving input source.
    fn verify(&mut self,_config:Value,_context:&Context)->Result<Value>{Err("native validation operation is not admitted".into())}
}
pub type Factory=fn(Value)->Result<Box<dyn Session>>;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Initialization {op:String,recipe_sha256:String,profiles:BTreeMap<String,Pin>,#[serde(default)] options:Value}
struct State {busy:bool,failed:bool,closed:bool,phase:String,request_id:Option<String>,result:Option<Value>,error:Option<String>,metadata:Value}
enum Work {Warm(PathBuf,mpsc::Sender<Reply>),Prefetch(PathBuf,usize,String,Option<Vec<String>>,mpsc::Sender<Reply>),Audit(mpsc::Sender<Reply>),
    WarmGated(PathBuf,PathBuf,PathBuf,PathBuf),
    Infer(PathBuf,PathBuf,PathBuf,Option<crate::phase_gate::Pin>),Verify(Value,Option<crate::phase_gate::Pin>),
    Evict(mpsc::Sender<Reply>),Stop(mpsc::Sender<Reply>)}
struct Actor {initialization_sha256:String,recipe_sha256:String,profiles:BTreeMap<String,Pin>,
    sender:mpsc::Sender<Work>,context:Context}
pub fn owned(mut value:Value)->Value {
    value["owner_pid"]=json!(std::process::id());
    value["owner_instance"]=json!(crate::process_identity::current());value
}
fn guarded(action:impl FnOnce()->Result<Value>)->Reply {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)) {
        Ok(value)=>value.map(owned).map_err(|e|e.to_string()),
        Err(_)=>Err("native backend panicked; retire owner".into()),
    }
}
impl Actor {
    fn create(config:Initialization,identity:String,factory:Factory)->std::result::Result<Self,String>{
        if config.op!="init" || config.profiles.is_empty() || config.profiles.len()>1024 || config.recipe_sha256.len()!=64 ||
            !config.recipe_sha256.bytes().all(|v|v.is_ascii_hexdigit()&&!v.is_ascii_uppercase()){
            return Err("invalid immutable backend initialization".into());
        }
        for pin in config.profiles.values(){pin.verify().map_err(|e|e.to_string())?;}
        let context=Context {state:Arc::new(Mutex::new(State {busy:false,failed:false,closed:false,phase:"initializing".into(),
            request_id:None,result:None,error:None,metadata:json!({})})),cancelled:Arc::new(AtomicBool::new(false)),decode_gate:None};
        let worker=context.clone();let(sender,receiver)=mpsc::channel();let(boot,ready)=mpsc::channel();
        thread::Builder::new().name("native-model".into()).spawn(move||{
            let mut session=match factory(config.options){Ok(s)=>s,Err(e)=>{let _=boot.send(Err(e.to_string()));return;}};
            let metadata=match session.metadata(){Ok(v)=>v,Err(e)=>{let _=boot.send(Err(e.to_string()));return;}};
            if let Ok(mut state)=worker.state.lock(){state.metadata=metadata;state.phase="idle".into();}
            if boot.send(Ok(())).is_err(){return;}
            while let Ok(work)=receiver.recv(){
                if let Work::Stop(reply)=work {
                    let result=guarded(||session.evict());drop(session);
                    if let Ok(mut state)=worker.state.lock(){state.busy=false;state.closed=true;state.phase="closed".into();state.result=None;}
                    let _=reply.send(result);return;
                }
                let inference=matches!(&work,Work::Infer(..)|Work::Verify(..));
                let warming=matches!(&work,Work::Warm(..)|Work::WarmGated(..));
                let mut completion=None;
                let(mut result,response)=match work {
                    Work::Warm(path,reply)=>(guarded(||{let value=session.warm(&path)?;crate::background_io::finish_pins()?;Ok(value)}),Some(reply)),
                    Work::WarmGated(path,gate,done,cancel)=>{
                        let result=guarded(||{
                            let io_gate=std::env::var_os("NATIVE_BACKGROUND_IO_GATE").map(PathBuf::from).unwrap_or_else(||gate.clone());
                            let paused=io_gate.with_extension("paused");
                            while !gate.exists(){
                                if cancel.exists(){return Err("background host warm cancelled".into());}
                                if paused.exists(){return Ok(json!({"paused":true}));}
                                thread::sleep(std::time::Duration::from_millis(2));
                            }
                            if cancel.exists(){return Err("background host warm cancelled".into());}
                            if paused.exists() || !io_gate.exists(){return Ok(json!({"paused":true}));}
                            let _scope=crate::background_io::Scope::enter(crate::background_io::Policy{gate:io_gate,cancel,stop:None,
                                pin_gate:std::env::var_os("NATIVE_BACKGROUND_PIN_GATE").map(PathBuf::from)});
                            match session.warm(&path).and_then(|value|{crate::background_io::finish_pins()?;Ok(value)}){
                                Err(error) if error.to_string()==crate::background_io::PAUSED=>Ok(json!({"paused":true})),
                                result=>result,
                            }
                        });
                        completion=Some(done);(result,None)
                    },
                    Work::Prefetch(path,budget,ticket,components,reply)=>{
                        (guarded(||{crate::background_io::finish_pins()?;let mut value=session.prefetch_components(&path,budget,components.as_deref())?;value["request_id"]=json!(ticket);Ok(value)}),Some(reply))
                    },
                    Work::Infer(profile,request,output,decode_gate)=>{
                        let context=Context{decode_gate,..worker.clone()};
                        (guarded(||session.infer(&profile,&request,&output,&context)),None)
                    },
                    Work::Verify(config,decode_gate)=>{
                        let context=Context{decode_gate,..worker.clone()};
                        (guarded(||session.verify(config,&context)),None)
                    },
                    Work::Evict(reply)=>(guarded(||session.evict()),Some(reply)),Work::Stop(_)=>unreachable!(),
                    Work::Audit(reply)=>(guarded(||session.audit()),Some(reply)),
                };
                // Paused preparation can retain real CPU allocations without
                // registration. Refresh metadata before publishing its yield;
                // metadata is observational and must not complete preparation.
                let metadata=guarded(||{let value=session.metadata()?;
                    if !value.is_object(){return Err("backend metadata must be an object".into());}Ok(value)});
                if let Err(error)=&metadata{
                    result=Err(match result{Err(original)=>format!("{original}; metadata refresh failed: {error}"),
                        Ok(_)=>format!("metadata refresh failed: {error}")});
                }
                if let (Ok(value),Ok(metadata))=(&mut result,&metadata){
                    if warming{
                        // warm() can return before framework-owned deferred
                        // registration finishes. Current metadata is the final
                        // resource snapshot; operation-only fields survive.
                        let paused=value.get("paused").cloned();
                        for(key,field)in metadata.as_object().unwrap(){value[key]=field.clone();}
                        if let Some(paused)=paused{value["paused"]=paused;}
                    }
                }
                if let Ok(mut state)=worker.state.lock(){
                    state.busy=false;
                    if let Err(error)=&result {state.failed=true;state.phase="failed".into();state.error=Some(error.clone());}
                    else{state.phase=if inference{"complete"}else{"idle"}.into();}
                    if inference{state.result=result.as_ref().ok().cloned();}
                    if let Ok(metadata)=metadata{state.metadata=metadata;}
                    // Publish while holding the admission mutex. A caller
                    // observing this file can reserve the now-idle actor;
                    // status cannot acknowledge a yield before its receipt.
                    if let Some(done)=completion{
                        let receipt=match &result{Ok(value)=>json!({"ok":true,"details":value}),
                            Err(error)=>json!({"ok":false,"error":error})};
                        let saved=(||->Result<()>{let temporary=done.with_extension("tmp");
                            std::fs::write(&temporary,serde_json::to_vec(&receipt)?)?;
                            std::fs::rename(temporary,done)?;Ok(())})();
                        if let Err(error)=saved{
                            state.failed=true;state.phase="failed".into();state.error=Some(error.to_string());
                        }
                    }
                }
                if let Some(reply)=response{let _=reply.send(result);}
            }
        }).map_err(|e|e.to_string())?;
        ready.recv().map_err(|e|e.to_string())??;
        Ok(Self{initialization_sha256:identity,recipe_sha256:config.recipe_sha256,profiles:config.profiles,sender,context})
    }
    fn profile(&self,input:&Value)->std::result::Result<PathBuf,String>{
        let pin=self.profiles.get(input["profile"].as_str().ok_or("profile label required")?).ok_or("profile is not admitted")?;
        pin.verify().map_err(|e|e.to_string())?;Ok(pin.path.clone())
    }
    fn status(&self)->Reply {
        let state=self.context.state.lock().map_err(|_|"backend state poisoned")?;
        Ok(owned(json!({"busy":state.busy,"failed":state.failed,"closed":state.closed,"phase":state.phase,
            "request_id":state.request_id,"result_ready":state.result.is_some(),"error":state.error,
            "metadata":state.metadata,"recipe_sha256":self.recipe_sha256})))
    }
    fn reserve(&self,ticket:Option<String>)->std::result::Result<(),String>{
        let mut state=self.context.state.lock().map_err(|_|"backend state poisoned")?;
        if state.failed || state.busy || state.closed{return Err("native backend is failed, busy or closed".into());}
        if ticket.is_some() && state.result.is_some(){return Err("acknowledge the previous result first".into());}
        state.busy=true;state.phase="admitted".into();
        if ticket.is_some(){state.request_id=ticket;self.context.cancelled.store(false,Ordering::Release);}Ok(())
    }
    fn synchronous(&self,work:impl FnOnce(mpsc::Sender<Reply>)->Work)->Reply {
        self.reserve(None)?;let(reply,result)=mpsc::channel();self.sender.send(work(reply)).map_err(|e|e.to_string())?;
        result.recv().map_err(|e|e.to_string())?
    }
    fn dispatch(&self,input:Value)->Reply {
        let ticket=||input["request_id"].as_str().filter(|v|!v.is_empty()).map(str::to_owned).ok_or("request ticket required");
        let gate=||input.get("decode_phase_gate").filter(|value|!value.is_null())
            .map(|value|serde_json::from_value::<crate::phase_gate::Pin>(value.clone()).map_err(|error|error.to_string())).transpose();
        match input["op"].as_str().ok_or("backend operation required")? {
            "status"=>self.status(),
            "warm"=>{let path=self.profile(&input)?;self.synchronous(|r|Work::Warm(path,r))},
            "warm-pause"=>{
                if !crate::weight_loader::flag("NATIVE_RESIDENT_BACKGROUND_HOST_LOAD").map_err(|e|e.to_string())?{
                    return Err("background host warm is not enabled".into());
                }
                // The controller has closed the IO gate. Wait only for the
                // current bounded read/pin to yield this actor, not for model
                // readiness. Partial reads remain available to foreground.
                loop{
                    let state=self.context.state.lock().map_err(|_|"backend state poisoned")?;
                    if !state.busy || state.phase!="host_warm"{break;}
                    drop(state);thread::sleep(std::time::Duration::from_millis(2));
                }
                self.status()
            },
            "warm-start"=>{
                if !crate::weight_loader::flag("NATIVE_RESIDENT_BACKGROUND_HOST_LOAD").map_err(|e|e.to_string())?{
                    return Err("background host warm is not enabled".into());
                }
                let path=self.profile(&input)?;
                let file=|key:&str|->std::result::Result<PathBuf,String>{
                    let value=PathBuf::from(input[key].as_str().ok_or("background host gate missing")?);
                    if !value.is_absolute() || value.components().any(|p|matches!(p,std::path::Component::ParentDir)){
                        return Err("absolute background host gate required".into());
                    }Ok(value)
                };
                let(gate,done,cancel)=(file("gate")?,file("done")?,file("cancel")?);
                self.reserve(None)?;
                self.context.phase("host_warm");
                self.sender.send(Work::WarmGated(path,gate,done,cancel)).map_err(|e|e.to_string())?;
                Ok(owned(json!({"started":true,"host_only":true,"gate":input["gate"]})))
            },
            "prefetch"=>{
                let path=self.profile(&input)?;let ticket=ticket()?;
                let budget=input["device_budget_bytes"].as_u64().and_then(|v|usize::try_from(v).ok()).ok_or("explicit device budget required")?;
                let components=input.get("preload_components").map(|value|->std::result::Result<Vec<String>,String>{
                    let names:Vec<String>=serde_json::from_value(value.clone()).map_err(|e|e.to_string())?;
                    let unique=names.iter().collect::<std::collections::BTreeSet<_>>();
                    if names.is_empty() || names.len()>1024 || unique.len()!=names.len() || names.iter().any(|n|n.is_empty()){
                        return Err("nonempty unique preload components required".into());
                    }Ok(names)
                }).transpose()?;
                let mut value=self.synchronous(|r|Work::Prefetch(path,budget,ticket,components,r))?;
                value["recipe_sha256"]=json!(self.recipe_sha256);Ok(value)
            },
            "start"=>{
                let profile=self.profile(&input)?;let ticket=ticket()?;
                let path=|key:&str|->std::result::Result<PathBuf,String>{let p=PathBuf::from(input[key].as_str().ok_or("internal path required")?);
                    if !p.is_absolute() || p.components().any(|v|matches!(v,std::path::Component::ParentDir)){return Err("absolute internal path required".into());}Ok(p)};
                let(request,output)=(path("request")?,path("output")?);let decode_gate=gate()?;self.reserve(Some(ticket.clone()))?;
                self.sender.send(Work::Infer(profile,request,output,decode_gate)).map_err(|e|e.to_string())?;
                Ok(owned(json!({"started":true,"request_id":ticket})))
            },
            "verify"=>{let ticket=ticket()?;let decode_gate=gate()?;self.reserve(Some(ticket.clone()))?;
                self.sender.send(Work::Verify(input["config"].clone(),decode_gate)).map_err(|e|e.to_string())?;
                Ok(owned(json!({"started":true,"request_id":ticket})))},
            "cancel"=>{
                let state=self.context.state.lock().map_err(|_|"backend state poisoned")?;
                if state.request_id.as_deref()!=Some(ticket()?.as_str()){return Err("cancellation ticket differs".into());}
                self.context.cancelled.store(true,Ordering::Release);Ok(owned(json!({"cancellation_requested":true})))
            },
            "result"|"ack"=>{
                let mut state=self.context.state.lock().map_err(|_|"backend state poisoned")?;
                if state.failed{return Err(state.error.clone().unwrap_or("native backend failed".into()));}
                if state.busy || state.request_id.as_deref()!=input["request_id"].as_str(){return Err("result ticket is not complete".into());}
                let result=state.result.as_ref().ok_or("native result missing")?.clone();
                if input["op"]=="ack"{state.result=None;state.request_id=None;state.phase="idle".into();return Ok(owned(json!({"acknowledged":true})));}
                Ok(owned(json!({"request_id":state.request_id,"report":result})))
            },
            "evict"=>self.synchronous(Work::Evict),"audit_weights"=>self.synchronous(Work::Audit),"shutdown"=>self.synchronous(Work::Stop),
            _=>Err("unported native backend operation".into()),
        }
    }
}
pub struct Backend {actor:Mutex<Option<Actor>>,factory:Factory,versions:fn()->Result<Value>}
impl Backend {
    pub const fn new(factory:Factory,versions:fn()->Result<Value>)->Self{Self{actor:Mutex::new(None),factory,versions}}
    pub fn dispatch(&self,input:&[u8])->Reply {
        let value:Value=serde_json::from_slice(input).map_err(|e|e.to_string())?;
        if value["op"]=="versions"{return guarded(self.versions);}
        let mut actor=self.actor.lock().map_err(|_|"backend admission poisoned")?;
        if value["op"]=="init"{
            let identity=format!("{:x}",Sha256::digest(input));
            if let Some(actor)=actor.as_ref(){if actor.initialization_sha256!=identity{return Err("backend initialization changed".into());}return actor.status();}
            let created=Actor::create(serde_json::from_value(value).map_err(|e|e.to_string())?,identity,self.factory)?;
            let result=created.status();*actor=Some(created);return result;
        }
        actor.as_ref().ok_or("native backend is not initialized")?.dispatch(value)
    }
    /// # Safety
    /// The caller follows backend_abi v1 and frees this allocation with `free`.
    pub unsafe fn call(&self,input:*const u8,bytes:usize,out:*mut Bytes)->i32 {
        if input.is_null() || out.is_null() || bytes==0 || bytes>backend_abi::MAX_REQUEST_BYTES{return -1;}
        out.write(Bytes{data:std::ptr::null_mut(),len:0});
        let result=std::panic::catch_unwind(||self.dispatch(std::slice::from_raw_parts(input,bytes)));
        let result=match result{Ok(v)=>v,Err(_)=>Err("native backend panicked; retire owner".into())};
        let status=if result.is_ok(){0}else{1};
        let value=match result{Ok(details)=>json!({"ok":true,"details":details}),Err(error)=>json!({"ok":false,"error":error})};
        let data=match serde_json::to_vec(&value){Ok(v)=>v.into_boxed_slice(),Err(_)=>return -1};
        if data.len()>backend_abi::MAX_REPLY_BYTES{return -1;}
        let len=data.len();out.write(Bytes{data:Box::into_raw(data).cast(),len});status
    }
}
/// # Safety
/// Must be the exact buffer returned by `Backend::call` in this DSO.
pub unsafe fn free(data:*mut u8,bytes:usize){if !data.is_null(){drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(data,bytes)));}}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fake;
    impl Session for Fake {
        fn metadata(&self)->Result<Value>{Ok(json!({"ready":true}))}
        fn warm(&mut self,_:&Path)->Result<Value>{Ok(json!({"device_bytes_uploaded":0}))}
        fn prefetch(&mut self,_:&Path,_:usize)->Result<Value>{Ok(json!({"copy_complete":true}))}
        fn evict(&mut self)->Result<Value>{Ok(json!({"evicted":true}))}
        fn infer(&mut self,_:&Path,request:&Path,_:&Path,context:&Context)->Result<Value>{
            context.phase("sampling");
            if request.ends_with("cancel") {loop{context.checkpoint()?;thread::yield_now();}}
            if request.ends_with("decode") {
                return Ok(json!({"computed":true,"decode_gate_completed":context.before_gpu_phase("decode",&[768,768,124])?.is_some()}));
            }
            Ok(json!({"computed":true}))
        }
    }
    fn fake(_:Value)->Result<Box<dyn Session>>{Ok(Box::new(Fake))}
    fn versions()->Result<Value>{Ok(json!({}))}
    fn initialized_with(factory:Factory,options:Value)->(Backend,PathBuf){
        static NEXT:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
        let file=std::env::temp_dir().join(format!("resident-{}-{}-{}",std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),NEXT.fetch_add(1,Ordering::Relaxed)));
        std::fs::write(&file,b"profile").unwrap();let backend=Backend::new(factory,versions);
        call(&backend,json!({"op":"init","recipe_sha256":"a".repeat(64),"profiles":{
            "one":{"path":file,"sha256":format!("{:x}",Sha256::digest(b"profile"))}},"options":options})).unwrap();
        (backend,file)
    }
    fn initialized()->(Backend,PathBuf){initialized_with(fake,json!({}))}
    fn call(backend:&Backend,value:Value)->Reply{backend.dispatch(&serde_json::to_vec(&value).unwrap())}
    fn complete(backend:&Backend)->Value{
        let limit=std::time::Instant::now()+std::time::Duration::from_secs(5);
        loop{let value=call(backend,json!({"op":"status"})).unwrap();if value["busy"]==false{return value;}
            assert!(std::time::Instant::now()<limit);thread::yield_now();}
    }
    fn completion(path:&Path)->Value{
        let limit=std::time::Instant::now()+std::time::Duration::from_secs(5);
        while !path.exists(){assert!(std::time::Instant::now()<limit);thread::yield_now();}
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }
    struct PausedHost {attempted:bool,metadata_error:bool,warm_error:bool}
    impl Session for PausedHost {
        fn metadata(&self)->Result<Value>{
            if self.attempted&&self.metadata_error{return Err("retained accounting unavailable".into());}
            Ok(json!({"host_resident_bytes":if self.attempted{8192}else{0},"host_pinned_bytes":0,"weights_ready":false}))
        }
        fn warm(&mut self,_:&Path)->Result<Value>{
            self.attempted=true;Err(if self.warm_error{"original warm guard failed"}else{crate::background_io::PAUSED}.into())
        }
        fn prefetch(&mut self,_:&Path,_:usize)->Result<Value>{unreachable!()}
        fn infer(&mut self,_:&Path,_:&Path,_:&Path,_:&Context)->Result<Value>{unreachable!()}
        fn evict(&mut self)->Result<Value>{Ok(json!({}))}
    }
    fn paused_host(options:Value)->Result<Box<dyn Session>>{Ok(Box::new(PausedHost{
        attempted:false,metadata_error:options["metadata_error"]==true,warm_error:options["warm_error"]==true}))}
    struct StaleWarm {completed:bool}
    impl Session for StaleWarm {
        fn metadata(&self)->Result<Value>{Ok(json!({"host_resident_bytes":8192,
            "host_pinned_bytes":if self.completed{8192}else{0},"profile_sha256":"current-residency"}))}
        fn warm(&mut self,_:&Path)->Result<Value>{
            self.completed=true;Ok(json!({"host_resident_bytes":8192,"host_pinned_bytes":0,
                "profile_sha256":"stale-snapshot","warmed_profile_sha256":"host-only-target",
                "host_weights_verified":true,"device_bytes_uploaded":0}))
        }
        fn prefetch(&mut self,_:&Path,_:usize)->Result<Value>{Ok(json!({"prefetch_only":true}))}
        fn infer(&mut self,_:&Path,_:&Path,_:&Path,_:&Context)->Result<Value>{Ok(json!({"infer_only":true}))}
        fn evict(&mut self)->Result<Value>{Ok(json!({}))}
    }
    fn stale_warm(_:Value)->Result<Box<dyn Session>>{Ok(Box::new(StaleWarm{completed:false}))}
    fn start_gated(backend:&Backend,file:&Path,gate:&Path,done:&Path){
        std::fs::write(gate,b"go").unwrap();
        let actor=backend.actor.lock().unwrap();let actor=actor.as_ref().unwrap();
        actor.reserve(None).unwrap();actor.context.phase("host_warm");
        actor.sender.send(Work::WarmGated(file.to_path_buf(),gate.to_path_buf(),done.to_path_buf(),file.with_extension("cancel"))).unwrap();
    }
    #[test]fn paused_warm_publishes_actual_retained_ram_without_pin_or_readiness_credit(){
        let(backend,file)=initialized_with(paused_host,json!({}));let gate=file.with_extension("gate");let done=file.with_extension("done");
        start_gated(&backend,&file,&gate,&done);let state=complete(&backend);let receipt=completion(&done);
        assert_eq!(state["failed"],false);assert_eq!(receipt["ok"],true);
        for details in [&state["metadata"],&receipt["details"]]{
            assert_eq!(details["host_resident_bytes"],8192);assert_eq!(details["host_pinned_bytes"],0);
            assert_eq!(details["weights_ready"],false);
        }
        assert_eq!(receipt["details"]["paused"],true);
        assert_eq!(receipt["details"]["owner_pid"],std::process::id());
        call(&backend,json!({"op":"shutdown"})).unwrap();
        for path in [file,gate,done]{std::fs::remove_file(path).unwrap();}
    }
    #[test]fn warm_and_gated_warm_receipts_use_final_metadata_and_preserve_operation_fields(){
        for gated in [false,true]{
            let(backend,file)=initialized_with(stale_warm,json!({}));
            let gate=file.with_extension("gate");let done=file.with_extension("done");
            let value=if gated{
                start_gated(&backend,&file,&gate,&done);complete(&backend);completion(&done)["details"].clone()
            }else{call(&backend,json!({"op":"warm","profile":"one"})).unwrap()};
            assert_eq!(value["host_pinned_bytes"],8192);assert_eq!(value["host_resident_bytes"],8192);
            assert_eq!(value["profile_sha256"],"current-residency");
            assert_eq!(value["warmed_profile_sha256"],"host-only-target");
            assert_eq!(value["host_weights_verified"],true);assert_eq!(value["device_bytes_uploaded"],0);
            assert_eq!(value["owner_pid"],std::process::id());assert!(!value["owner_instance"].as_str().unwrap().is_empty());
            // Metadata refresh is still observational for other operations.
            let prefetch=call(&backend,json!({"op":"prefetch","profile":"one","request_id":"copy","device_budget_bytes":1})).unwrap();
            assert_eq!(prefetch["prefetch_only"],true);assert!(prefetch.get("host_pinned_bytes").is_none());
            call(&backend,json!({"op":"start","profile":"one","request_id":"infer","request":"/request","output":"/output"})).unwrap();
            complete(&backend);let result=call(&backend,json!({"op":"result","request_id":"infer"})).unwrap();
            assert_eq!(result["report"]["infer_only"],true);assert!(result["report"].get("host_pinned_bytes").is_none());
            assert_eq!(call(&backend,json!({"op":"ack","request_id":"infer"})).unwrap()["acknowledged"],true);
            call(&backend,json!({"op":"shutdown"})).unwrap();
            std::fs::remove_file(file).unwrap();if gated{std::fs::remove_file(gate).unwrap();std::fs::remove_file(done).unwrap();}
        }
    }
    #[test]fn metadata_refresh_failure_is_fatal_and_preserves_an_original_warm_failure(){
        for warm_error in [false,true]{
            let(backend,file)=initialized_with(paused_host,json!({"metadata_error":true,"warm_error":warm_error}));
            let gate=file.with_extension("gate");let done=file.with_extension("done");start_gated(&backend,&file,&gate,&done);
            let state=complete(&backend);let receipt=completion(&done);
            assert_eq!(state["failed"],true);assert_eq!(receipt["ok"],false);
            let error=receipt["error"].as_str().unwrap();assert!(error.contains("retained accounting unavailable"));
            assert_eq!(error.contains("original warm guard failed"),warm_error);
            // Failed owners reject fresh admission, including shutdown. The
            // test owns the worker and explicitly joins its stop response.
            let actor=backend.actor.lock().unwrap();let actor=actor.as_ref().unwrap();
            let(send,receive)=mpsc::channel();actor.sender.send(Work::Stop(send)).unwrap();receive.recv().unwrap().unwrap();
            for path in [file,gate,done]{std::fs::remove_file(path).unwrap();}
        }
    }
    #[test]fn invalid_component_selection_cannot_reserve_the_native_actor(){
        let(backend,file)=initialized();
        for names in [json!([]),json!(["dit","dit"]),json!([""]),json!(null),json!([1])]{
            assert!(call(&backend,json!({"op":"prefetch","profile":"one","request_id":"target",
                "device_budget_bytes":1024,"preload_components":names})).is_err());
            let state=complete(&backend);assert_eq!(state["failed"],false);assert_eq!(state["busy"],false);
        }
        call(&backend,json!({"op":"shutdown"})).unwrap();std::fs::remove_file(file).unwrap();
    }
    #[test]fn unsupported_component_preload_errors_instead_of_loading_the_whole_model(){
        let(backend,file)=initialized();
        let error=call(&backend,json!({"op":"prefetch","profile":"one","request_id":"target",
            "device_budget_bytes":1024,"preload_components":["dit"]})).unwrap_err();
        assert!(error.contains("does not admit component preload"));
        assert_eq!(complete(&backend)["failed"],true);
        std::fs::remove_file(file).unwrap();
    }
    #[test]fn gated_host_warm_does_not_start_before_its_gate_and_publishes_a_receipt(){
        let(backend,file)=initialized();let gate=file.with_extension("gate");
        let done=file.with_extension("done");let cancel=file.with_extension("cancel");
        {
            let actor=backend.actor.lock().unwrap();let actor=actor.as_ref().unwrap();
            actor.reserve(None).unwrap();actor.context.phase("host_warm");
            actor.sender.send(Work::WarmGated(file.clone(),gate.clone(),done.clone(),cancel.clone())).unwrap();
        }
        thread::sleep(std::time::Duration::from_millis(20));assert!(!done.exists());
        let pause=gate.with_extension("paused");std::fs::write(&pause,b"").unwrap();
        assert_eq!(complete(&backend)["failed"],false);
        let paused=completion(&done);
        assert_eq!(paused["details"]["paused"],true);
        // A closed gate yields the actor, so ordinary foreground work can run.
        assert_eq!(call(&backend,json!({"op":"warm","profile":"one"})).unwrap()["device_bytes_uploaded"],0);
        std::fs::remove_file(&done).unwrap();
        std::fs::remove_file(&pause).unwrap();
        std::fs::write(&gate,b"go").unwrap();
        {
            let actor=backend.actor.lock().unwrap();let actor=actor.as_ref().unwrap();
            actor.reserve(None).unwrap();actor.context.phase("host_warm");
            actor.sender.send(Work::WarmGated(file.clone(),gate.clone(),done.clone(),cancel.clone())).unwrap();
        }
        assert_eq!(complete(&backend)["failed"],false);
        let receipt=completion(&done);
        assert_eq!(receipt["ok"],true);assert_eq!(receipt["details"]["device_bytes_uploaded"],0);
        call(&backend,json!({"op":"shutdown"})).unwrap();
        for path in [file,gate,done]{std::fs::remove_file(path).unwrap();}
    }
    #[test]fn cancelling_a_closed_host_gate_joins_and_reports_failure(){
        let(backend,file)=initialized();let gate=file.with_extension("gate");
        let done=file.with_extension("done");let cancel=file.with_extension("cancel");
        std::fs::write(&cancel,b"stop").unwrap();
        {
            let actor=backend.actor.lock().unwrap();let actor=actor.as_ref().unwrap();
            actor.reserve(None).unwrap();actor.context.phase("host_warm");
            actor.sender.send(Work::WarmGated(file.clone(),gate.clone(),done.clone(),cancel.clone())).unwrap();
        }
        assert_eq!(complete(&backend)["failed"],true);
        let receipt=completion(&done);assert_eq!(receipt["ok"],false);
        for path in [file,done,cancel]{std::fs::remove_file(path).unwrap();}
    }
    #[test]fn result_stays_bound_to_ticket_until_acknowledged(){
        let(backend,file)=initialized();
        call(&backend,json!({"op":"start","profile":"one","request_id":"first","request":"/request","output":"/output"})).unwrap();
        assert_eq!(complete(&backend)["result_ready"],true);
        assert!(call(&backend,json!({"op":"result","request_id":"other"})).is_err());
        assert!(call(&backend,json!({"op":"start","profile":"one","request_id":"second","request":"/request","output":"/output"})).is_err());
        assert_eq!(call(&backend,json!({"op":"result","request_id":"first"})).unwrap()["report"]["computed"],true);
        call(&backend,json!({"op":"ack","request_id":"first"})).unwrap();
        call(&backend,json!({"op":"shutdown"})).unwrap();std::fs::remove_file(file).unwrap();
    }
    #[test]fn cancelled_work_is_joined_before_owner_is_reported_failed(){
        let(backend,file)=initialized();
        call(&backend,json!({"op":"start","profile":"one","request_id":"first","request":"/cancel","output":"/output"})).unwrap();
        assert!(call(&backend,json!({"op":"cancel","request_id":"other"})).is_err());
        call(&backend,json!({"op":"cancel","request_id":"first"})).unwrap();
        let value=complete(&backend);assert_eq!(value["failed"],true);assert_eq!(value["result_ready"],false);
        assert!(call(&backend,json!({"op":"prefetch","profile":"one","request_id":"second","device_budget_bytes":1024})).is_err());
        std::fs::remove_file(file).unwrap();
    }
    #[test]fn changed_profile_cannot_start_dma_or_inference(){
        let(backend,file)=initialized();std::fs::write(&file,b"other").unwrap();
        assert!(call(&backend,json!({"op":"warm","profile":"one"})).is_err());
        assert!(call(&backend,json!({"op":"prefetch","profile":"one","request_id":"first","device_budget_bytes":1024})).is_err());
        assert_eq!(call(&backend,json!({"op":"status"})).unwrap()["busy"],false);
        call(&backend,json!({"op":"shutdown"})).unwrap();std::fs::remove_file(file).unwrap();
    }
    #[test]fn standard_actor_scopes_the_decode_gate_to_one_request(){
        use std::{io::{BufRead,BufReader,Write},os::unix::net::UnixListener};
        let(backend,file)=initialized();let socket=file.with_extension("sock");
        let listener=UnixListener::bind(&socket).unwrap();
        let peer=thread::spawn(move||{
            let(mut stream,_)=listener.accept().unwrap();let mut line=String::new();
            BufReader::new(&mut stream).read_line(&mut line).unwrap();
            let request:Value=serde_json::from_str(&line).unwrap();
            assert_eq!(request["stage"],"decode");assert_eq!(request["geometry"],json!([768,768,124]));
            stream.write_all(&serde_json::to_vec(&json!({"schema":"native-decode-phase-gate-v1",
                "token":"a".repeat(32),"continued":true})).unwrap()).unwrap();
        });
        call(&backend,json!({"op":"start","profile":"one","request_id":"first","request":"/decode","output":"/output",
            "decode_phase_gate":{"schema":"native-decode-phase-gate-v1","path":socket,"token":"a".repeat(32),
                "stage":"decode","geometry":[768,768,124]}})).unwrap();
        assert_eq!(complete(&backend)["failed"],false);peer.join().unwrap();
        assert_eq!(call(&backend,json!({"op":"result","request_id":"first"})).unwrap()["report"]["decode_gate_completed"],true);
        call(&backend,json!({"op":"ack","request_id":"first"})).unwrap();
        std::fs::remove_file(socket).unwrap();
        call(&backend,json!({"op":"start","profile":"one","request_id":"second","request":"/decode","output":"/output"})).unwrap();
        assert_eq!(complete(&backend)["failed"],false);
        assert_eq!(call(&backend,json!({"op":"result","request_id":"second"})).unwrap()["report"]["decode_gate_completed"],false);
        call(&backend,json!({"op":"ack","request_id":"second"})).unwrap();
        call(&backend,json!({"op":"shutdown"})).unwrap();std::fs::remove_file(file).unwrap();
    }
    #[test]fn cancellation_during_phase_submission_prevents_gpu_decode(){
        use std::{io::{BufRead,BufReader,Write},os::unix::net::UnixListener};
        let(backend,file)=initialized();let socket=file.with_extension("sock");
        let listener=UnixListener::bind(&socket).unwrap();
        let(ready,waiting)=mpsc::channel();let(release,continued)=mpsc::channel();
        let peer=thread::spawn(move||{
            let(mut stream,_)=listener.accept().unwrap();let mut line=String::new();
            BufReader::new(&mut stream).read_line(&mut line).unwrap();ready.send(()).unwrap();
            continued.recv().unwrap();
            stream.write_all(&serde_json::to_vec(&json!({"schema":"native-decode-phase-gate-v1",
                "token":"a".repeat(32),"continued":true})).unwrap()).unwrap();
        });
        call(&backend,json!({"op":"start","profile":"one","request_id":"first","request":"/decode","output":"/output",
            "decode_phase_gate":{"schema":"native-decode-phase-gate-v1","path":socket,"token":"a".repeat(32),
                "stage":"decode","geometry":[768,768,124]}})).unwrap();
        waiting.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        call(&backend,json!({"op":"cancel","request_id":"first"})).unwrap();release.send(()).unwrap();
        let status=complete(&backend);assert_eq!(status["failed"],true);assert_eq!(status["result_ready"],false);
        peer.join().unwrap();std::fs::remove_file(socket).unwrap();std::fs::remove_file(file).unwrap();
    }
}
