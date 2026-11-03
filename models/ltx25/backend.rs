//! Native worker ABI: one thread-owned session, no subprocess or Python calls.
use crate::{gold::sha,request::Session};
use native_core_infra::{backend_abi::{self,Bytes},native_backend::Pin};
use serde::Deserialize;
use serde_json::{json,Value};
use std::{collections::BTreeMap,path::PathBuf,sync::{mpsc,Arc,Mutex,OnceLock},thread};
type Reply=std::result::Result<Value,String>;
unsafe extern "C" {
    fn cudaRuntimeGetVersion(value:*mut i32)->i32;
    fn cublasGetProperty(property:i32,value:*mut i32)->i32;
    fn cudnnGetVersion()->usize;
    fn nvrtcVersion(major:*mut i32,minor:*mut i32)->i32;
}
fn versions()->Reply{
    let(mut runtime,mut major,mut minor,mut nv_major,mut nv_minor)=(0,0,0,0,0);
    let statuses=unsafe{[cudaRuntimeGetVersion(&mut runtime),cublasGetProperty(0,&mut major),
        cublasGetProperty(1,&mut minor),nvrtcVersion(&mut nv_major,&mut nv_minor)]};
    if statuses.iter().any(|v|*v!=0){return Err("native runtime version query failed".into());}
    Ok(owned(json!({"cuda_runtime":runtime,"cublas_major":major,"cublas_minor":minor,
        "cudnn":unsafe{cudnnGetVersion()},"nvrtc":[nv_major,nv_minor]})))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Initialization {op:String,recipe_sha256:String,profiles:BTreeMap<String,Pin>,
    #[serde(default)] checkpoint:Option<native_core_infra::checkpoint::Source>,
    #[serde(default)] options:Options}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {world_size:usize}
impl Default for Options {fn default()->Self{Self{world_size:1}}}
struct State {busy:bool,failed:bool,closed:bool,phase:String,request_id:Option<String>,result:Option<Value>,error:Option<String>,metadata:Value}
enum Work {
    Warm(PathBuf,mpsc::Sender<Reply>),
    Prefetch(PathBuf,usize,String,Option<Vec<String>>,mpsc::Sender<Reply>),
    Infer(PathBuf,PathBuf,PathBuf,Option<u64>,Option<native_core_infra::phase_gate::Pin>),
    Verify(crate::request::Config,Option<native_core_infra::phase_gate::Pin>),
    Evict(mpsc::Sender<Reply>),
    Stop(mpsc::Sender<Reply>),
}
struct Actor {initialization_sha256:String,recipe_sha256:String,profiles:BTreeMap<String,Pin>,
    sender:mpsc::Sender<Work>,state:Arc<Mutex<State>>}
fn owned(mut value:Value)->Value{
    value["owner_pid"]=json!(std::process::id());
    value["owner_instance"]=json!(native_core_infra::process_identity::current());value
}
fn guarded(action:impl FnOnce()->crate::Result<Value>)->Reply{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)){
        Ok(result)=>result.map(owned).map_err(|error|error.to_string()),
        Err(_)=>Err("native backend panicked; retire owner".into()),
    }
}
impl Actor {
    fn create(config:Initialization,identity:String)->std::result::Result<Self,String>{
        if config.op!="init" || !matches!(config.options.world_size,1|2) || config.profiles.is_empty() || config.profiles.len()>32 ||
            config.recipe_sha256.len()!=64 || !config.recipe_sha256.bytes().all(|v|v.is_ascii_hexdigit() && !v.is_ascii_uppercase()){
            return Err("invalid immutable backend initialization".into());
        }
        for pin in config.profiles.values(){pin.verify().map_err(|e|e.to_string())?;}
        // Admission fails before spawning the CUDA owner. A missing pin must
        // not silently select the diagnostic full-file compatibility loader.
        let checkpoint=crate::admission::checkpoint(config.checkpoint.as_ref())?;
        let state=Arc::new(Mutex::new(State{busy:false,failed:false,closed:false,phase:"initializing".into(),request_id:None,
            result:None,error:None,metadata:json!({})}));
        let worker_state=state.clone();let (sender,receiver)=mpsc::channel();let (boot,ready)=mpsc::channel();
        thread::Builder::new().name("native-ltx25".into()).spawn(move||{
            let mut session=match Session::open_backend_world(checkpoint,config.options.world_size){
                Ok(session)=>session,
                Err(error)=>{let _=boot.send(Err(error.to_string()));return;}
            };
            if let Some(source)=&config.checkpoint{if let Err(error)=session.set_checkpoint_identity(source.sha256.clone()){
                let _=boot.send(Err(error.to_string()));return;
            }}
            let callback_state=worker_state.clone();
            session.set_phase_callback(Box::new(move|phase|{
                if let Ok(mut state)=callback_state.lock(){state.phase=phase.into();}
            }));
            let metadata=match session.metadata(){Ok(value)=>value,Err(error)=>{let _=boot.send(Err(error.to_string()));return;}};
            if let Ok(mut state)=worker_state.lock(){state.metadata=metadata.clone();state.phase="idle".into();}
            if boot.send(Ok(metadata)).is_err(){return;}
            while let Ok(work)=receiver.recv(){
                let work=match work{
                    Work::Stop(reply)=>{
                        let result=guarded(||{session.evict()?;Ok(json!({"closed":true}))});
                        drop(session);
                        if let Ok(mut state)=worker_state.lock(){state.busy=false;state.closed=true;state.phase="closed".into();state.result=None;}
                        let _=reply.send(result);return;
                    },work=>work,
                };
                let inference=matches!(&work,Work::Infer(..)|Work::Verify(..));
                let (result,response)=match work{
                    Work::Warm(profile,reply)=>(guarded(||session.warm_host(&profile)),Some(reply)),
                    Work::Prefetch(profile,budget,request_id,components,reply)=>{
                        let result=guarded(||{let mut value=if let Some(components)=components{
                            session.prefetch_components(&profile,budget,&components)?
                        }else{session.prefetch(&profile,budget)?};
                            value["request_id"]=json!(request_id);Ok(value)});(result,Some(reply))
                    },
                    Work::Infer(profile,request,output,generation,gate)=>(guarded(||session.infer_prepared(&profile,&request,&output,generation,gate)),None),
                    Work::Verify(config,gate)=>(guarded(||session.run_with_gate(config,gate)),None),
                    Work::Evict(reply)=>(guarded(||{session.evict()?;session.metadata()}),Some(reply)),
                    Work::Stop(_)=>unreachable!(),
                };
                if let Ok(mut state)=worker_state.lock(){
                    state.busy=false;
                    if let Err(error)=&result{state.failed=true;state.error=Some(error.clone());state.phase="failed".into();}
                    else{state.phase=if inference{"complete"}else{"idle"}.into();}
                    if inference{state.result=result.as_ref().ok().cloned();}
                    if let Ok(metadata)=session.metadata(){state.metadata=metadata;}
                }
                if let Some(response)=response{let _=response.send(result);}
            }
        }).map_err(|e|e.to_string())?;
        ready.recv().map_err(|e|e.to_string())??;
        Ok(Self{initialization_sha256:identity,recipe_sha256:config.recipe_sha256,
            profiles:config.profiles,sender,state})
    }
    fn profile(&self,value:&Value)->std::result::Result<PathBuf,String>{
        let label=value["profile"].as_str().ok_or("profile label required")?;
        let pin=self.profiles.get(label).ok_or("profile is not admitted")?;
        pin.verify().map_err(|e|e.to_string())?;Ok(pin.path.clone())
    }
    fn status(&self)->Reply{
        let state=self.state.lock().map_err(|_|"backend state poisoned")?;
        Ok(owned(json!({"busy":state.busy,"failed":state.failed,"closed":state.closed,"phase":state.phase,
            "request_id":state.request_id,"result_ready":state.result.is_some(),"error":state.error,
            "metadata":state.metadata,"recipe_sha256":self.recipe_sha256})))
    }
    fn reserve(&self,request_id:Option<String>)->std::result::Result<(),String>{
        let mut state=self.state.lock().map_err(|_|"backend state poisoned")?;
        if state.failed || state.busy || state.closed{return Err("native backend is failed, busy or closed".into());}
        if request_id.is_some() && state.result.is_some(){return Err("acknowledge the previous result first".into());}
        state.busy=true;state.phase="admitted".into();
        if request_id.is_some(){state.request_id=request_id;}
        Ok(())
    }
    fn synchronous(&self,work:impl FnOnce(mpsc::Sender<Reply>)->Work)->Reply{
        self.reserve(None)?;let (reply,result)=mpsc::channel();
        self.sender.send(work(reply)).map_err(|e|e.to_string())?;
        result.recv().map_err(|e|e.to_string())?
    }
    fn dispatch(&self,input:Value)->Reply{
        match input["op"].as_str().ok_or("backend operation required")?{
            "status"=>self.status(),
            "warm"=>{let profile=self.profile(&input)?;self.synchronous(|reply|Work::Warm(profile,reply))},
            "prefetch"=>{
                let profile=self.profile(&input)?;
                let budget=input["device_budget_bytes"].as_u64().and_then(|v|usize::try_from(v).ok()).ok_or("explicit device budget required")?;
                let request_id=input["request_id"].as_str().filter(|v|!v.is_empty()).ok_or("prefetch ticket required")?.to_owned();
                let components=input.get("preload_components").map(|v|serde_json::from_value::<Vec<String>>(v.clone())
                    .map_err(|e|format!("preload component selection: {e}"))).transpose()?;
                let mut result=self.synchronous(|reply|Work::Prefetch(profile,budget,request_id,components,reply))?;
                result["recipe_sha256"]=json!(self.recipe_sha256);Ok(result)
            },
            "start"=>{
                let profile=self.profile(&input)?;
                let path=|key:&str|->std::result::Result<PathBuf,String>{let path=PathBuf::from(input[key].as_str().ok_or("request/output path required")?);
                    if !path.is_absolute(){return Err("absolute internal request/output paths required".into());}Ok(path)};
                let (request,output)=(path("request")?,path("output")?);
                let request_id=input["request_id"].as_str().filter(|v|!v.is_empty()).ok_or("request identity required")?.to_owned();
                let generation=input.get("prepared_generation").map(|value|value.as_u64().ok_or("invalid prepared generation")).transpose()?;
                let gate=input.get("decode_phase_gate").filter(|value|!value.is_null())
                    .map(|value|serde_json::from_value(value.clone()).map_err(|error|error.to_string())).transpose()?;
                self.reserve(Some(request_id.clone()))?;
                self.sender.send(Work::Infer(profile,request,output,generation,gate)).map_err(|e|e.to_string())?;
                Ok(owned(json!({"started":true,"request_id":request_id})))
            },
            "verify"=>{
                let config=serde_json::from_value(input["config"].clone()).map_err(|error|error.to_string())?;
                let request_id=input["request_id"].as_str().filter(|v|!v.is_empty()).ok_or("request identity required")?.to_owned();
                let gate=input.get("decode_phase_gate").filter(|value|!value.is_null())
                    .map(|value|serde_json::from_value(value.clone()).map_err(|error|error.to_string())).transpose()?;
                self.reserve(Some(request_id.clone()))?;
                self.sender.send(Work::Verify(config,gate)).map_err(|e|e.to_string())?;
                Ok(owned(json!({"started":true,"request_id":request_id})))
            },
            "result"|"ack"=>{
                let mut state=self.state.lock().map_err(|_|"backend state poisoned")?;
                if state.failed{return Err(state.error.clone().unwrap_or("native backend failed".into()));}
                if state.busy || state.request_id.as_deref()!=input["request_id"].as_str(){return Err("result ticket is not complete".into());}
                let result=state.result.as_ref().ok_or("completed native result missing")?.clone();
                if input["op"]=="ack"{state.result=None;state.request_id=None;state.phase="idle".into();return Ok(owned(json!({"acknowledged":true})));}
                Ok(owned(json!({"request_id":state.request_id,"report":result})))
            },
            "evict"=>self.synchronous(Work::Evict),
            "shutdown"=>self.synchronous(Work::Stop),
            _=>Err("unported native backend operation".into()),
        }
    }
}
fn dispatch(input:&[u8])->Reply{
    static ACTOR:OnceLock<Mutex<Option<Actor>>>=OnceLock::new();
    let value:Value=serde_json::from_slice(input).map_err(|e|e.to_string())?;
    if value["op"]=="versions"{return versions();}
    let mut actor=ACTOR.get_or_init(||Mutex::new(None)).lock().map_err(|_|"backend admission poisoned")?;
    if value["op"]=="init"{
        let identity=sha(input);
        if let Some(actor)=actor.as_ref(){
            if actor.initialization_sha256!=identity{return Err("backend initialization changed".into());}return actor.status();
        }
        let initialized=Actor::create(serde_json::from_value(value).map_err(|e|e.to_string())?,identity)?;
        let result=initialized.status();*actor=Some(initialized);return result;
    }
    actor.as_ref().ok_or("native backend is not initialized")?.dispatch(value)
}
#[no_mangle]
pub extern "C" fn native_backend_version_v1()->u32{backend_abi::VERSION}
/// # Safety
/// input/output buffers follow backend_abi v1; free only via the matching DSO.
#[no_mangle]
pub unsafe extern "C" fn native_backend_call_v1(input:*const u8,bytes:usize,out:*mut Bytes)->i32{
    if input.is_null() || out.is_null() || bytes==0 || bytes>backend_abi::MAX_REQUEST_BYTES{return -1;}
    out.write(Bytes{data:std::ptr::null_mut(),len:0});
    let result=std::panic::catch_unwind(||dispatch(std::slice::from_raw_parts(input,bytes)));
    let result=match result{Ok(value)=>value,Err(_)=>Err("native backend panicked; retire owner".into())};
    let status=if result.is_ok(){0}else{1};
    let value=match result{Ok(details)=>json!({"ok":true,"details":details}),Err(error)=>json!({"ok":false,"error":error})};
    let bytes=serde_json::to_vec(&value).unwrap_or_else(|_|br#"{"ok":false,"error":"native serialization failed"}"#.to_vec()).into_boxed_slice();
    if bytes.len()>backend_abi::MAX_REPLY_BYTES{return -1;}
    let len=bytes.len();out.write(Bytes{data:Box::into_raw(bytes).cast(),len});status
}
/// # Safety
/// This exact allocation must have been returned by native_backend_call_v1.
#[no_mangle]
pub unsafe extern "C" fn native_backend_free_v1(data:*mut u8,bytes:usize){
    if !data.is_null(){drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(data,bytes)));}
}
