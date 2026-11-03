//! Request preparation overlaps readers and CUDA creation. Model code supplies
//! CPU preprocessing and backend preparation; scheduling and timing are shared.
use std::{ffi::{c_int,c_void},sync::mpsc,thread,time::{SystemTime,UNIX_EPOCH}};
type Result<T> = std::result::Result<T,String>;
unsafe extern "C" {
    fn cuCtxGetCurrent(context:*mut *mut c_void)->c_int;
    fn cuCtxSetCurrent(context:*mut c_void)->c_int;
}
fn now()->f64{SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()}

pub struct Prepared<T>{pub value:T,pub cpu_ready_at:f64,pub backend_ready_at:f64}
pub struct Preparation<T>{gate:Option<mpsc::Sender<usize>>,worker:Option<thread::JoinHandle<Result<Prepared<T>>>>}
impl<T:Send+'static> Preparation<T>{
    /// Starts CPU work immediately, before the caller creates a CUDA context.
    /// Backend work waits until `device_ready` and uses the caller's current
    /// context, including a private context. That context must outlive this job.
    pub fn start(cpu:impl FnOnce()->Result<T>+Send+'static,
                 backend:impl FnOnce(&T)->Result<()>+Send+'static)->Result<Self>{
        crate::startup_trace::configure_from_env();
        let (tx,rx)=mpsc::channel();
        let worker=thread::Builder::new().name("startup-prepare".into()).spawn(move||{
            let value=crate::startup_trace::measure("startup","cpu_prepare",0,cpu)?;
            let cpu_ready_at=now();
            let context=rx.recv().map_err(|_|"startup preparation cancelled")?;
            let code=unsafe{cuCtxSetCurrent(context as *mut c_void)};
            if code!=0{return Err(format!("preparation cuCtxSetCurrent: {code}"));}
            crate::startup_trace::measure("startup","backend_prepare",0,||backend(&value))?;
            Ok(Prepared{value,cpu_ready_at,backend_ready_at:now()})
        }).map_err(|e|format!("preparation thread: {e}"))?;
        Ok(Self{gate:Some(tx),worker:Some(worker)})
    }
    pub fn device_ready(&mut self)->Result<()>{
        let mut context=std::ptr::null_mut();
        let code=unsafe{cuCtxGetCurrent(&mut context)};
        if code!=0 || context.is_null(){return Err(format!("preparation needs a live CUDA context: {code}"));}
        // A closed receiver means CPU preparation already returned an error;
        // finish() joins it and preserves that original diagnostic.
        let _=self.gate.take().ok_or("preparation device already released")?.send(context as usize);
        Ok(())
    }
    pub fn finish(mut self)->Result<Prepared<T>>{
        self.gate.take();
        self.worker.take().unwrap().join().map_err(|_|"preparation thread panicked")?
    }
}
impl<T> Drop for Preparation<T>{
    fn drop(&mut self){
        // A cancelled caller closes the gate and joins: no detached work may
        // touch a context after its owner has started teardown.
        self.gate.take();
        if let Some(worker)=self.worker.take(){let _=worker.join();}
    }
}
