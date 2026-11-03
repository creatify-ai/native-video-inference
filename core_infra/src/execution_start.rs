//! Opt-in first-model execution boundary, timestamped by the CUDA stream.
//!
//! The host callback runs after all preceding stream dependencies and directly
//! before the first model operation. It does no CUDA work and adds no device
//! synchronization. Each callback owns its timestamp storage until completion,
//! even if the request fails or its observer goes away.
use crate::Result;
use serde_json::{json,Value};
use std::{cell::RefCell,sync::{Arc,OnceLock,atomic::{AtomicU64,Ordering}}};
use std::{fs::{File,OpenOptions},os::fd::AsRawFd};
use std::os::unix::net::UnixDatagram;
static SIGNAL:OnceLock<Option<UnixDatagram>>=OnceLock::new();
fn configure_signal(){
    SIGNAL.get_or_init(||{
        let path=std::env::var_os("NATIVE_FIRST_DIT_SIGNAL_PATH")?;
        let socket=UnixDatagram::unbound().ok()?;
        socket.set_nonblocking(true).ok()?;socket.connect(path).ok()?;Some(socket)
    });
}

#[repr(C)] struct Timespec {sec:i64,nsec:i64}
unsafe extern "C" {fn clock_gettime(clock:i32,out:*mut Timespec)->i32;}
unsafe extern "C" {fn write(fd:i32,data:*const std::ffi::c_void,bytes:usize)->isize;fn getpid()->i32;fn pause()->i32;}
struct Notification {file:Option<File>,hold:bool}
static NOTIFY:OnceLock<Notification>=OnceLock::new();
fn configure_notification(){
    NOTIFY.get_or_init(||{
        let file=std::env::var_os("NATIVE_FIRST_DIT_NOTIFY_FILE").and_then(|p|
            match OpenOptions::new().create(true).append(true).open(p){Ok(f)=>Some(f),Err(e)=>{
                eprintln!("first DiT notification: {e}");None}});
        let hold=file.is_some() && std::env::var("NATIVE_FIRST_DIT_HOLD").as_deref()==Ok("1");
        Notification{file,hold}
    });
}
/// The clock used by native stream markers. Request admission can record its
/// start in this same domain without a CUDA synchronization or a second clock.
/// A zero value means the platform clock could not be read.
/// Host preparation timestamps share the clock used by native execution
/// markers, so separate CUDA owners can report comparable copy intervals.
pub fn monotonic_ns()->u64{
    let mut time=Timespec{sec:0,nsec:0};
    if unsafe{clock_gettime(1,&mut time)}!=0 || time.sec<0 || time.nsec<0{return 0;}
    (time.sec as u64).saturating_mul(1_000_000_000).saturating_add(time.nsec as u64)
}
/// A named phase's stream boundary, without first-DiT notifications or holds.
/// The callback retains its storage independently of the observer.
pub struct StreamTimestamp(Arc<AtomicU64>);
unsafe extern "C" fn stamp_phase(data:*mut std::ffi::c_void){
    let time=unsafe{Arc::from_raw(data.cast::<AtomicU64>())};time.store(monotonic_ns(),Ordering::Release);
}
impl StreamTimestamp{
    /// # Safety
    /// `stream` is live on the caller's current CUDA context.
    pub unsafe fn record(stream:*mut std::ffi::c_void)->Result<Self>{
        let time=Arc::new(AtomicU64::new(0));
        #[cfg(feature="cuda")]{
            let raw=Arc::into_raw(time.clone()).cast_mut().cast();
            let code=unsafe{cudaLaunchHostFunc(stream,Some(stamp_phase),raw)};
            if code!=0{return Err(format!("phase timestamp enqueue failed: {code}").into());}
            Ok(Self(time))
        }
        #[cfg(not(feature="cuda"))]{let _=(stream,time);Err("phase timestamp requires CUDA".into())}
    }
    /// Read after ordinary phase completion, not by synchronizing in a callback.
    pub fn read(&self)->Result<u64>{let ns=self.0.load(Ordering::Acquire);if ns==0{Err("phase stream marker incomplete".into())}else{Ok(ns)}}
}
fn enabled()->bool{
    static ENABLED:OnceLock<bool>=OnceLock::new();
    *ENABLED.get_or_init(||std::env::var("NATIVE_FIRST_DIT_TRACE").as_deref()==Ok("1"))
}
#[derive(Clone)]
struct State {time:Arc<AtomicU64>,armed:bool,enqueued:bool}
thread_local! {static ACTIVE:RefCell<Option<State>>=const{RefCell::new(None)};}
pub struct Capture {time:Option<Arc<AtomicU64>>,root:bool}
impl Capture {
    pub fn begin()->Self{
        if !enabled(){return Self{time:None,root:false};}
        configure_notification();
        ACTIVE.with(|slot|{
            let mut slot=slot.borrow_mut();
            if let Some(state)=slot.as_ref(){return Self{time:Some(state.time.clone()),root:false};}
            let time=Arc::new(AtomicU64::new(0));
            *slot=Some(State{time:time.clone(),armed:false,enqueued:false});
            Self{time:Some(time),root:true}
        })
    }
    /// Read only after the caller's normal completion synchronization.
    pub fn report(&self)->Result<Option<Value>>{
        let Some(time)=&self.time else{return Ok(None);};
        let ns=time.load(Ordering::Acquire);
        if ns==0{return Err("first DiT stream marker did not complete".into());}
        Ok(Some(json!({"monotonic_ns":ns,"clock":"CLOCK_MONOTONIC",
            "owner_pid":std::process::id(),"owner_instance":crate::process_identity::current(),
            "scope":"CUDA stream callback immediately before the first DiT operation, after preceding stream dependencies"})))
    }
}
impl Drop for Capture {
    fn drop(&mut self){if self.root{ACTIVE.with(|slot|{slot.borrow_mut().take();});}}
}
/// Arm inside the model forward, after its input/weight preparation. The next
/// hooked operation enqueues the marker immediately before its actual launch.
pub fn arm(){ACTIVE.with(|slot|{if let Some(state)=slot.borrow_mut().as_mut(){
    if !state.enqueued{state.armed=true;}
}});}

#[cfg(feature="cuda")]
unsafe extern "C" {
    fn cudaLaunchHostFunc(stream:*mut std::ffi::c_void,
        callback:Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,data:*mut std::ffi::c_void)->i32;
}
unsafe extern "C" fn stamp(data:*mut std::ffi::c_void){
    // CUDA owns this Arc lease after a successful enqueue. Never call CUDA
    // APIs, allocate, acquire a mutex, or unwind from the callback.
    let time=unsafe{Arc::from_raw(data.cast::<AtomicU64>())};
    let ns=monotonic_ns();time.store(ns,Ordering::Release);
    if let Some(Some(signal))=SIGNAL.get(){
        let mut packet=[0u8;16];
        packet[..8].copy_from_slice(&u64::from(std::process::id()).to_le_bytes());
        packet[8..].copy_from_slice(&ns.to_le_bytes());
        let _=signal.send(&packet);
    }
    if let Some(notification)=NOTIFY.get(){
        if let Some(file)=&notification.file{
            // Fixed-size, allocation-free notification of the SAME timestamp
            // returned by report(). The observer can retire the whole owner
            // tree without waiting for denoising or decoding to finish.
            let message=[unsafe{getpid()} as u64,ns];
            unsafe{write(file.as_raw_fd(),message.as_ptr().cast(),std::mem::size_of_val(&message));}
        }
        // Opt-in iteration mode: keep subsequent GPU work behind this marker
        // until the external observer kills the process. No CUDA call, Rust
        // mutex, allocation or destructor runs in the callback.
        if notification.hold{loop{unsafe{pause();}}}
    }
}
pub fn enqueue_if_armed(stream:*mut std::ffi::c_void)->Result<()>{
    let time=ACTIVE.with(|slot|{
        let mut slot=slot.borrow_mut();let state=slot.as_mut()?;
        if !state.armed || state.enqueued{return None;}
        state.armed=false;state.enqueued=true;Some(state.time.clone())
    });
    let Some(time)=time else{return Ok(());};
    configure_signal();
    #[cfg(feature="cuda")]
    {
        let data=Arc::into_raw(time).cast_mut().cast();
        let code=unsafe{cudaLaunchHostFunc(stream,Some(stamp),data)};
        if code!=0{
            // CUDA can surface an earlier asynchronous error here. Retain
            // the raw lease until callback completion or owner retirement;
            // an indeterminate enqueue must not create a dangling callback.
            return Err(format!("first DiT cudaLaunchHostFunc failed: {code}").into());
        }
        Ok(())
    }
    #[cfg(not(feature="cuda"))]
    {let _=(stream,time);Err("first DiT marker requires CUDA".into())}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]fn phase_callback_owns_storage_after_observer_drop(){
        let time=Arc::new(AtomicU64::new(0));let observed=time.clone();
        let marker=StreamTimestamp(time.clone());let callback=Arc::into_raw(time).cast_mut().cast();drop(marker);
        unsafe{stamp_phase(callback);}
        assert!(observed.load(Ordering::Acquire)>0);assert_eq!(Arc::strong_count(&observed),1);
    }
    #[test] fn callback_retains_storage_after_capture_drop(){
        let time=Arc::new(AtomicU64::new(0));let observed=time.clone();
        let callback=Arc::into_raw(time).cast_mut().cast();
        unsafe{stamp(callback);}
        assert!(observed.load(Ordering::Acquire)>0);
        assert_eq!(Arc::strong_count(&observed),1);
    }
    #[test] fn state_is_thread_local(){
        ACTIVE.with(|slot|*slot.borrow_mut()=Some(State{time:Arc::new(AtomicU64::new(7)),armed:true,enqueued:false}));
        std::thread::spawn(||ACTIVE.with(|slot|assert!(slot.borrow().is_none()))).join().unwrap();
        ACTIVE.with(|slot|{assert_eq!(slot.borrow().as_ref().unwrap().time.load(Ordering::Relaxed),7);slot.borrow_mut().take();});
    }
}
