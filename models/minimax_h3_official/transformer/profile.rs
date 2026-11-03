//! Diagnostic stream intervals, independently switchable between requests.
//! These are not CUPTI kernel-active timings. Selected children are disjoint;
//! their parent forward interval overlaps them and must not be added to them.
use std::{collections::BTreeMap, sync::{Mutex,OnceLock,atomic::{AtomicBool,AtomicI32,Ordering}},time::Instant};
use serde_json::{json,Value};

static ENABLED:AtomicBool=AtomicBool::new(false);
static STEP:AtomicI32=AtomicI32::new(-1);
#[derive(Default)]
struct Aggregate {count:u64,seconds:f64,max_seconds:f64,bytes:u64}
type Key=(i32,&'static str);
#[derive(Default)]
struct State {
    request:usize,mode_file:Option<String>,host:BTreeMap<Key,Aggregate>,gpu:BTreeMap<Key,Aggregate>,
}
static STATE:OnceLock<Mutex<State>>=OnceLock::new();
fn state()->&'static Mutex<State>{STATE.get_or_init(||Mutex::new(State::default()))}
pub fn enabled()->bool{ENABLED.load(Ordering::Relaxed)}
pub fn set_step(step:i32){STEP.store(step,Ordering::Relaxed);}

pub fn begin_request(request:usize)->crate::Result<bool>{
    #[cfg(feature="cuda")]
    device::assert_drained()?;
    let path=std::env::var("H3_DIT_PROFILE_MODE_FILE").ok();
    let active=match &path {
        None=>false,
        Some(path)=>match std::fs::read_to_string(path)?.trim(){
            "0"=>false,"1"=>true,_=>return Err("profile mode file must contain 0 or 1".into()),
        },
    };
    *state().lock().unwrap()=State{request,mode_file:path,..State::default()};
    STEP.store(-1,Ordering::Relaxed);ENABLED.store(active,Ordering::Relaxed);
    Ok(active)
}

fn add(map:&mut BTreeMap<Key,Aggregate>,step:i32,label:&'static str,seconds:f64,bytes:usize){
    let row=map.entry((step,label)).or_default();row.count+=1;row.seconds+=seconds;
    row.max_seconds=row.max_seconds.max(seconds);row.bytes+=bytes as u64;
}
pub fn record(label:&'static str,start:Instant,bytes:usize){
    if enabled(){add(&mut state().lock().unwrap().host,STEP.load(Ordering::Relaxed),label,start.elapsed().as_secs_f64(),bytes);}
}
pub struct HostSpan {row:Option<(Instant,i32,&'static str,usize)>}
impl HostSpan {
    pub fn new(label:&'static str,bytes:usize)->Self{
        Self{row:if enabled(){Some((Instant::now(),STEP.load(Ordering::Relaxed),label,bytes))}else{None}}
    }
}
impl Drop for HostSpan {
    fn drop(&mut self){if let Some((started,step,label,bytes))=self.row {
        add(&mut state().lock().unwrap().host,step,label,started.elapsed().as_secs_f64(),bytes);
    }}
}

#[cfg(feature="cuda")]
mod device {
    use super::*;
    use std::{ffi::c_void,ptr};
    unsafe extern "C" {
        fn cudaEventCreate(event:*mut *mut c_void)->i32;
        fn cudaEventRecord(event:*mut c_void,stream:*mut c_void)->i32;
        fn cudaEventSynchronize(event:*mut c_void)->i32;
        fn cudaEventElapsedTime(ms:*mut f32,start:*mut c_void,end:*mut c_void)->i32;
        fn cudaEventDestroy(event:*mut c_void)->i32;
    }
    struct Pending {step:i32,label:&'static str,start:usize,end:usize}
    static PENDING:OnceLock<Mutex<Vec<Pending>>>=OnceLock::new();
    static ERRORS:OnceLock<Mutex<Vec<String>>>=OnceLock::new();
    fn pending()->&'static Mutex<Vec<Pending>>{PENDING.get_or_init(||Mutex::new(vec![]))}
    fn errors()->&'static Mutex<Vec<String>>{ERRORS.get_or_init(||Mutex::new(vec![]))}
    fn check(code:i32)->crate::Result<()>{
        if code!=0{return Err(format!("profiling CUDA event error {code}").into());}Ok(())
    }
    pub fn assert_drained()->crate::Result<()>{
        if !pending().lock().unwrap().is_empty()||!errors().lock().unwrap().is_empty(){
            return Err("cannot change profiling mode with pending events or an event error".into());
        }Ok(())
    }
    pub struct GpuSpan {row:Option<Pending>,stream:*mut c_void}
    impl GpuSpan {
        pub fn new(label:&'static str,stream:*mut c_void)->crate::Result<Self>{
            if !enabled(){return Ok(Self{row:None,stream});}
            let timer=Instant::now();let mut start=ptr::null_mut();let mut end=ptr::null_mut();
            unsafe{
                check(cudaEventCreate(&mut start))?;
                if let Err(error)=check(cudaEventCreate(&mut end)){cudaEventDestroy(start);return Err(error);}
                if let Err(error)=check(cudaEventRecord(start,stream)){
                    cudaEventDestroy(start);cudaEventDestroy(end);return Err(error);
                }
            }
            record("instrumentation.create_record_start",timer,0);
            Ok(Self{row:Some(Pending{step:STEP.load(Ordering::Relaxed),label,start:start as usize,end:end as usize}),stream})
        }
    }
    impl Drop for GpuSpan {
        fn drop(&mut self){if let Some(row)=self.row.take(){
            let timer=Instant::now();let code=unsafe{cudaEventRecord(row.end as *mut c_void,self.stream)};
            if code!=0{errors().lock().unwrap().push(format!("end event {}: {code}",row.label));}
            pending().lock().unwrap().push(row);record("instrumentation.record_stop",timer,0);
        }}
    }
    pub fn collect()->crate::Result<()>{
        let rows=std::mem::take(&mut *pending().lock().unwrap());
        let mut failure=None;
        for row in rows {
            let mut ms=0f32;
            let results=unsafe{[
                cudaEventSynchronize(row.end as *mut c_void),
                cudaEventElapsedTime(&mut ms,row.start as *mut c_void,row.end as *mut c_void),
                cudaEventDestroy(row.start as *mut c_void),cudaEventDestroy(row.end as *mut c_void),
            ]};
            if let Some(code)=results.into_iter().find(|&code|code!=0){failure=Some(format!("profile event error {code}"));}
            else {add(&mut state().lock().unwrap().gpu,row.step,row.label,ms as f64/1000.0,0);}
        }
        let errors=errors().lock().unwrap();
        if !errors.is_empty(){return Err(format!("profile event errors: {errors:?}").into());}
        if let Some(error)=failure{return Err(error.into());}Ok(())
    }
}
#[cfg(feature="cuda")]
pub use device::GpuSpan;

pub fn collect_step()->crate::Result<()>{
    if !enabled(){return Ok(());}
    let timer=Instant::now();
    #[cfg(feature="cuda")]
    device::collect()?;
    record("instrumentation.collect_events",timer,0);Ok(())
}

pub fn finish()->crate::Result<Value>{
    collect_step()?;
    let state=state().lock().unwrap();
    let rows=|map:&BTreeMap<Key,Aggregate>|map.iter().map(|((step,label),r)|
        json!({"step":step,"label":label,"count":r.count,"seconds":r.seconds,"bytes":r.bytes,"max_seconds":r.max_seconds})).collect::<Vec<_>>();
    Ok(json!({"schema":"h3-resident-dit-profile-v1","enabled":enabled(),"request_index":state.request,
        "mode_file":state.mode_file,"host":rows(&state.host),"cuda_stream_intervals":rows(&state.gpu),
        "step_labels":"-1 request setup/reset; 0..5 denoising; 6 final raw output",
        "cuda_definition":"CUDA event intervals on the original stream; include submission gaps and backend dependencies; not CUPTI kernel-active time",
        "host_definition":"Instant wall durations. Forward/step parents overlap children; host and device records are separate clocks",
        "collection":"after each original step endpoint D2H/sync; no new per-operation stream synchronization",
        "disjoint_forward_children":["dense_attention.main","dense_attention.refiner","sparse_attention","router","ffn.main","ffn.refiner"],
        "other_definition":"derived forward CUDA interval minus disjoint selected children; includes QKV/output projections, norm, RoPE, modulation/residual, packing, gaps and unclassified work",
        "measurement_policy":"profile-off creates no CUDA events; compare ON request wall against adjacent OFF requests, not summed component intervals"}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aggregation_preserves_counts_bytes_and_max(){
        let mut map=BTreeMap::new();add(&mut map,2,"test",0.2,8);add(&mut map,2,"test",0.3,16);
        let row=&map[&(2,"test")];assert_eq!(row.count,2);assert_eq!(row.bytes,24);
        assert_eq!(row.seconds,0.5);assert_eq!(row.max_seconds,0.3);
    }
}
