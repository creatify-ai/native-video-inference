//! Opt-in timing spans only; no CUDA synchronization or changed data path.
use std::{fs::{File,OpenOptions},io::Write,path::{Path,PathBuf},sync::{Mutex,OnceLock,atomic::{AtomicBool,Ordering}}};
#[repr(C)] struct Timespec {sec:i64,nsec:i64}
unsafe extern "C" {fn clock_gettime(clock:i32,out:*mut Timespec)->i32;}
fn clock(id:i32)->f64{let mut t=Timespec{sec:0,nsec:0};unsafe{clock_gettime(id,&mut t);}t.sec as f64+t.nsec as f64*1e-9}
#[derive(Clone,Default)]
pub struct Configuration {pub directory:Option<PathBuf>,pub stop_file:Option<PathBuf>}
static CONFIG:OnceLock<Configuration>=OnceLock::new();
/// Configure diagnostics before the first measured operation.
pub fn configure(config:Configuration) {let _=CONFIG.set(config);}
/// Shared loaders enable the same diagnostics for every model. An absent
/// variable leaves explicit model configuration available.
pub fn configure_from_env() {
    if let Some(directory)=std::env::var_os("NATIVE_STARTUP_TRACE_DIR") {
        configure(Configuration{directory:Some(directory.into()),stop_file:None});
    }
}
/// An opt-in host timestamp; this does not synchronize the CUDA stream.
pub fn timestamp()->Option<f64>{CONFIG.get().and_then(|c|c.directory.as_ref()).map(|_|clock(1))}
static FILE:OnceLock<Option<Mutex<File>>>=OnceLock::new();
static STOP:OnceLock<Option<PathBuf>>=OnceLock::new();
static STOPPED:AtomicBool=AtomicBool::new(false);
fn file()->Option<&'static Mutex<File>>{FILE.get_or_init(||CONFIG.get().and_then(|c|c.directory.clone()).map(|dir|Mutex::new(OpenOptions::new().create(true).append(true).open(Path::new(&dir).join(format!("rust-{}.jsonl",std::process::id()))).expect("trace open")))).as_ref()}
fn enabled()->bool{if CONFIG.get().and_then(|c|c.directory.as_ref()).is_none(){return false;}if STOPPED.load(Ordering::Relaxed){return false;}if STOP.get_or_init(||CONFIG.get().and_then(|c|c.stop_file.clone())).as_ref().is_some_and(|p|p.exists()){STOPPED.store(true,Ordering::Relaxed);return false;}file().is_some()}
fn selected(kind:&str)->bool{
    static PHASES:OnceLock<bool>=OnceLock::new();
    if !*PHASES.get_or_init(||matches!(std::env::var("NATIVE_STARTUP_TRACE_LEVEL").as_deref(),Ok("phases"|"io"))){return true;}
    static IO:OnceLock<bool>=OnceLock::new();
    let io=*IO.get_or_init(||std::env::var("NATIVE_STARTUP_TRACE_LEVEL").as_deref()==Ok("io"));
    (io&&matches!(kind,"reader_metadata"|"reader_seek"|"read_file"|"direct_file_read"|"reader_guard_finish"|"reader_backpressure"|"upload_wait_reader"|"h2d_api"|"copy_wait"))||kind.starts_with("declared_")||kind.starts_with("startup_")||matches!(kind,
        "model_load"|"model_upload"|"model_allocate"|"model_replicate"|"text_encoder"|"tokenizer_prepare"|"backend_plan_prepare"|"pinned_register"|"upload_queue")
}
pub struct Span(Option<(&'static str,String,usize,f64,f64)>);
/// A cross-thread wall interval. Thread CPU clocks cannot be subtracted across
/// owners; the explicit flag keeps its zero placeholder out of CPU accounting.
pub struct AsyncSpan(Option<(&'static str,String,f64)>);
impl AsyncSpan {
    pub fn new(kind:&'static str,label:&str)->Self {
        Self(if selected(kind)&&enabled(){Some((kind,label.into(),clock(1)))}else{None})
    }
}
impl Drop for AsyncSpan {
    fn drop(&mut self) {
        if let Some((kind,label,start))=self.0.take() {
            let event=serde_json::json!({"kind":kind,"path":label,"bytes":0,"start":start,"end":clock(1),
                "cpu_s":0.,"cpu_measured":false,"pid":std::process::id(),"thread":"async"});
            let mut encoded=serde_json::to_vec(&event).expect("trace JSON");encoded.push(b'\n');
            file().unwrap().lock().unwrap().write_all(&encoded).expect("trace write");
        }
    }
}
impl Span{pub fn new(kind:&'static str,label:impl AsRef<str>,bytes:usize)->Self{Self(if selected(kind)&&enabled(){Some((kind,label.as_ref().to_owned(),bytes,clock(1),clock(3)))}else{None})}}
impl Drop for Span{fn drop(&mut self){if let Some((kind,label,bytes,start,cpu))=self.0.take(){let event=serde_json::json!({"kind":kind,"path":label,"bytes":bytes,"start":start,"end":clock(1),"cpu_s":clock(3)-cpu,"pid":std::process::id(),"thread":format!("{:?}",std::thread::current().id())});let mut encoded=serde_json::to_vec(&event).expect("trace JSON");encoded.push(b'\n');file().unwrap().lock().unwrap().write_all(&encoded).expect("trace write");}}}
pub fn measure<T>(kind:&'static str,label:impl AsRef<str>,bytes:usize,f:impl FnOnce()->T)->T{let _span=Span::new(kind,label,bytes);f()}
pub fn read(path:impl AsRef<Path>)->std::io::Result<Vec<u8>>{let path=path.as_ref();let mut span=Span::new("read_file",path.to_string_lossy(),0);let out=std::fs::read(path);if let (Some((_,_,bytes,_,_)),Ok(data))=(&mut span.0,&out){*bytes=data.len();}out}
