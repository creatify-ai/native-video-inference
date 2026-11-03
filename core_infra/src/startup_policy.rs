//! One serving policy for native component adapters. Model code declares the
//! preparation work; this module owns reader defaults and CPU/init overlap.
use crate::weight_loader::Config;
type Result<T> = std::result::Result<T,String>;

pub fn enabled()->bool{std::env::var("NATIVE_STARTUP").as_deref()==Ok("1")}
fn number(name:&str,default:usize,low:usize,high:usize)->Result<usize>{
    let value=match std::env::var(name){Ok(v)=>v.parse().map_err(|_|format!("invalid {name}"))?,Err(_)=>default};
    if !(low..=high).contains(&value){return Err(format!("{name} must be {low}..={high}"));}Ok(value)
}
pub fn readers()->Result<Config>{readers_with_default_workers(24)}
pub fn readers_with_default_workers(default:usize)->Result<Config>{
    if !(1..=64).contains(&default){return Err("reader default must be 1..=64".into());}
    let workers=number("NATIVE_LOAD_WORKERS",default,1,64)?;
    let chunk_bytes=number("NATIVE_LOAD_CHUNK_MIB",32,1,128)?<<20;
    let pinned_slots=number("NATIVE_LOAD_PINNED_SLOTS",4,1,16)?;
    let needed=(2*workers+pinned_slots)*chunk_bytes;
    let budget_bytes=number("NATIVE_LOAD_BUDGET_MIB",needed.div_ceil(1<<20),1,65536)?<<20;
    let config=Config{workers,chunk_bytes,pinned_slots,budget_bytes};config.validate()?;Ok(config)
}
pub fn direct()->Result<bool>{
    match std::env::var("NATIVE_LOAD_DIRECT").as_deref(){Err(_)|Ok("1")=>Ok(true),Ok("0")=>Ok(false),
        _=>Err("NATIVE_LOAD_DIRECT must be 0 or 1".into())}
}
pub fn light()->Result<bool>{
    match std::env::var("NATIVE_LOAD_VERIFY").as_deref(){Err(_)|Ok("full")=>Ok(false),Ok("light")=>Ok(true),
        _=>Err("NATIVE_LOAD_VERIFY must be full or light".into())}
}

pub struct CpuPrepared<T>{pub value:T,pub seconds:f64}
pub struct CpuPreparation<T>{worker:Option<std::thread::JoinHandle<Result<CpuPrepared<T>>>>}
impl<T:Send+'static> CpuPreparation<T>{
    /// Immutable CPU assets only. No CUDA state crosses this boundary.
    pub fn start(name:&'static str,prepare:impl FnOnce()->Result<T>+Send+'static)->Result<Self>{
        let worker=std::thread::Builder::new().name(name.into()).spawn(move||{
            let started=std::time::Instant::now();
            let value=crate::startup_trace::measure("startup_cpu_prepare",name,0,prepare)?;
            Ok(CpuPrepared{value,seconds:started.elapsed().as_secs_f64()})
        }).map_err(|e|e.to_string())?;
        Ok(Self{worker:Some(worker)})
    }
    /// Required preparation publishes a value only after its model validator
    /// accepts the actual result. The worker is joined on every path.
    pub fn finish_validated(self,validate:impl FnOnce(&T)->Result<()>)->Result<CpuPrepared<T>>{
        let prepared=self.finish()?;validate(&prepared.value)?;Ok(prepared)
    }
    pub fn finish(mut self)->Result<CpuPrepared<T>>{
        self.worker.take().unwrap().join().map_err(|_|"CPU preparation panicked")?
    }
}
impl<T> Drop for CpuPreparation<T>{
    fn drop(&mut self){if let Some(worker)=self.worker.take(){let _=worker.join();}}
}

/// CPU-only work starts before foreground initialization. The foreground may
/// own !Send CUDA handles. Join CPU work on every success/error path before the
/// caller can destroy the foreground's resources; no detached tasks remain.
pub fn overlap<T:Send,D>(cpu:impl FnOnce()->Result<T>+Send,
    foreground:impl FnOnce()->Result<D>)->Result<(T,D)>{
    std::thread::scope(|scope|{
        let job=std::thread::Builder::new().name("native-cpu-prepare".into())
            .spawn_scoped(scope,cpu).map_err(|e|e.to_string())?;
        let initialized=foreground();
        let prepared=job.join().map_err(|_|"native CPU preparation panicked")?;
        Ok((prepared?,initialized?))
    })
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn required_preparation_rejects_a_false_receipt_after_join(){
        let pending=CpuPreparation::start("required-result",||Ok(false)).unwrap();
        let error=pending.finish_validated(|value|if *value{Ok(())}else{Err("required preparation did not complete".into())}).err().unwrap();
        assert_eq!(error,"required preparation did not complete");
    }
    #[test]fn required_preparation_keeps_the_original_worker_error(){
        let pending=CpuPreparation::start("required-error",||Err::<bool,_>("layout failed".into())).unwrap();
        let error=pending.finish_validated(|_|panic!("failed work reached readiness validation")).err().unwrap();
        assert_eq!(error,"layout failed");
    }
    #[test]fn joins_cpu_on_foreground_failure(){
        let complete=std::sync::atomic::AtomicBool::new(false);
        let result=overlap(||{complete.store(true,std::sync::atomic::Ordering::Release);Ok(7)},||Err::<(),_>("device failed".into()));
        assert_eq!(result.unwrap_err(),"device failed");assert!(complete.load(std::sync::atomic::Ordering::Acquire));
    }
    #[test]fn foreground_can_own_thread_local_resources(){
        let (cpu,owner)=overlap(||Ok(7),||Ok(std::rc::Rc::new(9))).unwrap();
        assert_eq!(cpu,7);assert_eq!(*owner,9);
    }
    #[test]fn cpu_preparation_is_joined_on_cancellation(){
        let complete=std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));let worker=complete.clone();
        let pending=CpuPreparation::start("test-cpu-prepare",move||{
            worker.store(true,std::sync::atomic::Ordering::Release);Ok(())}).unwrap();
        drop(pending);assert!(complete.load(std::sync::atomic::Ordering::Acquire));
    }
}

/// The declared service owns release of optional next-model host preparation.
/// This does not weaken selected-model payload guards or copy admission.
pub fn background_preparation_deferred()->Result<bool> {
    match std::env::var("NATIVE_BACKGROUND_PREPARATION_DEFERRED").as_deref() {
        Err(std::env::VarError::NotPresent)|Ok("0")=>Ok(false),Ok("1")=>Ok(true),
        _=>Err("NATIVE_BACKGROUND_PREPARATION_DEFERRED must be 0 or 1".into()),
    }
}
