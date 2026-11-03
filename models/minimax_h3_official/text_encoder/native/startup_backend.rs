// One owned, metadata-only preparation of the first request's actual shape.
use std::sync::{Arc, atomic::AtomicBool};
use std::thread::JoinHandle;

unsafe extern "C" {
    fn cudaGetDevice(device:*mut c_int)->c_int;
    fn q3_attention_prepare_isolated(device:c_int,vision:i64,text:i64,out:*mut *mut c_void)->c_int;
    fn q3_attention_adopt_isolated(prepared:*mut c_void)->c_int;
    fn q3_attention_drop_isolated(prepared:*mut c_void);
    fn q3_conv_prepare_isolated(device:c_int,rows:i64,out:*mut *mut c_void)->c_int;
    fn q3_conv_adopt_isolated(destination:*mut c_void,prepared:*mut c_void)->c_int;
}

struct PreparedBackend {
    attention:usize,conv:usize,vision:usize,text:usize,images:usize,seconds:f64,
}
impl Drop for PreparedBackend {
    fn drop(&mut self){unsafe{
        if self.attention!=0{q3_attention_drop_isolated(self.attention as *mut c_void);}
        if self.conv!=0{h3_conv_destroy(self.conv as *mut c_void);}
    }}
}
struct StartupPreparation {
    path:PathBuf,stop:Arc<AtomicBool>,thread:Option<JoinHandle<std::result::Result<Option<PreparedBackend>,String>>>,
}
impl StartupPreparation {
    fn start()->Result<Option<Self>> {
        let Some(path)=std::env::var_os("H3_QWEN_STARTUP_HINT_FILE").map(PathBuf::from) else{return Ok(None)};
        if !path.is_absolute(){return Err("Qwen startup hint must be absolute".into());}
        let mut device=0;cuda(unsafe{cudaGetDevice(&mut device)})?;
        let stop=Arc::new(AtomicBool::new(false));let closed=stop.clone();let hint=path.clone();
        let thread=std::thread::Builder::new().name("qwen-first-shape-prepare".into()).spawn(move|| {
            let work=||->Result<Option<PreparedBackend>> {
                while !hint.is_file(){
                    if closed.load(Ordering::Relaxed){return Ok(None);}
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                if closed.load(Ordering::Relaxed){return Ok(None);}
                if fs::metadata(&hint)?.len()>8192{return Err("Qwen startup hint too large".into());}
                let value:Value=serde_json::from_slice(&fs::read(&hint)?)?;
                if value["schema"]!="h3-qwen-startup-shape-v1"{return Err("Qwen startup hint schema".into());}
                let get=|key:&str|->Result<usize>{usize::try_from(value[key].as_u64().ok_or("Qwen startup shape field")?).map_err(Into::into)};
                let (vision,text,images)=(get("vision_sequence")?,get("text_rows")?,get("images")?);
                if images>2 || text==0 || text>262144 || vision>262144 || (images>0 && vision==0){
                    return Err("Qwen startup shape exceeds metadata budget".into());
                }
                let started=Instant::now();
                let mut prepared=PreparedBackend{attention:0,conv:0,vision:if images>0{vision}else{0},text,images,seconds:0.0};
                let mut attention=ptr::null_mut();
                native_error(unsafe{q3_attention_prepare_isolated(device,prepared.vision as i64,text as i64,&mut attention)},q3_attention_error)?;
                prepared.attention=attention as usize;
                if images>0 {
                    let mut conv=ptr::null_mut();
                    native_error(unsafe{q3_conv_prepare_isolated(device,(images*vision) as i64,&mut conv)},h3_conv_error)?;
                    prepared.conv=conv as usize;
                }
                prepared.seconds=started.elapsed().as_secs_f64();
                Ok(Some(prepared))
            };
            work().map_err(|e|e.to_string())
        })?;
        Ok(Some(Self{path,stop,thread:Some(thread)}))
    }
    fn requested(&self)->bool{self.path.is_file()}
    fn adopt(mut self,conv:*mut c_void,cache:&mut crate::plan_cache::Cache)->Result<Value>{
        let waited=Instant::now();
        let Some(mut prepared)=self.thread.take().ok_or("Qwen preparation already joined")?
            .join().map_err(|_|"Qwen preparation worker panicked")?.map_err(|e|format!("Qwen backend preparation: {e}"))?
            else{return Ok(json!({"prepared":false,"forward_executed":false}))};
        // All startup shapes fit the three slots reserved by the verified
        // preload cache. Register both attention and convolution identities.
        cache.register_prepared_images(prepared.vision,prepared.text,prepared.images)?;
        let attention=std::mem::replace(&mut prepared.attention,0) as *mut c_void;
        native_error(unsafe{q3_attention_adopt_isolated(attention)},q3_attention_error)?;
        if prepared.conv!=0 {
            let ready=std::mem::replace(&mut prepared.conv,0) as *mut c_void;
            native_error(unsafe{q3_conv_adopt_isolated(conv,ready)},h3_conv_error)?;
        }
        Ok(json!({"prepared":true,"vision_sequence":prepared.vision,"text_rows":prepared.text,
            "images":prepared.images,"preparation_seconds":prepared.seconds,
            "join_seconds":waited.elapsed().as_secs_f64(),"forward_executed":false,"graphs_launched":0}))
    }
}
impl Drop for StartupPreparation {
    fn drop(&mut self){
        self.stop.store(true,Ordering::Relaxed);
        if let Some(thread)=self.thread.take(){let _=thread.join();}
    }
}

impl ConditionerServer {
    fn adopt_startup_backend(&mut self)->Result<Option<Value>> {
        if self.requests!=0{return Ok(None);}
        let Some(model)=self.model.as_mut() else{return Ok(None)};
        let Some(prepared)=model.startup_backend.take() else{return Ok(None)};
        // A first request without a supported hint uses the ordinary builder;
        // its idle metadata worker must not survive to another request shape.
        if !prepared.requested(){drop(prepared);return Ok(None);}
        self.in_gpu=true;
        Ok(Some(prepared.adopt(model.conv,&mut self.cache)?))
    }
}
