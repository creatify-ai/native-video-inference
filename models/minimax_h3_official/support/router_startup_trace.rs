//! Router crate observer without extra crate dependencies. Labels are literals.
use std::{fs::{File,OpenOptions},io::Write,path::{Path,PathBuf},sync::{Mutex,OnceLock,atomic::{AtomicBool,Ordering}}};
#[repr(C)] struct Timespec{sec:i64,nsec:i64}
unsafe extern "C"{fn clock_gettime(clock:i32,out:*mut Timespec)->i32;}
fn clock(id:i32)->f64{let mut t=Timespec{sec:0,nsec:0};unsafe{clock_gettime(id,&mut t);}t.sec as f64+t.nsec as f64*1e-9}
static FILE:OnceLock<Option<Mutex<File>>>=OnceLock::new();static STOP:OnceLock<Option<PathBuf>>=OnceLock::new();static STOPPED:AtomicBool=AtomicBool::new(false);
fn file()->Option<&'static Mutex<File>>{FILE.get_or_init(||std::env::var("H3_STARTUP_TRACE_DIR").ok().map(|dir|Mutex::new(OpenOptions::new().create(true).append(true).open(Path::new(&dir).join(format!("rust-{}.jsonl",std::process::id()))).expect("router trace open")))).as_ref()}
pub struct Span(Option<(&'static str,&'static str,usize,f64,f64)>);
impl Span{pub fn new(kind:&'static str,label:&'static str,bytes:usize)->Self{if STOPPED.load(Ordering::Relaxed){return Self(None);}if STOP.get_or_init(||std::env::var("H3_STARTUP_TRACE_STOP_FILE").ok().map(PathBuf::from)).as_ref().is_some_and(|p|p.exists()){STOPPED.store(true,Ordering::Relaxed);return Self(None);}Self(if file().is_some(){Some((kind,label,bytes,clock(1),clock(3)))}else{None})}}
impl Drop for Span{fn drop(&mut self){if let Some((kind,label,bytes,start,cpu))=self.0.take(){let line=format!("{{\"kind\":{kind:?},\"path\":{label:?},\"bytes\":{bytes},\"start\":{start:.9},\"end\":{:.9},\"cpu_s\":{:.9},\"pid\":{},\"thread\":\"{:?}\"}}\n",clock(1),clock(3)-cpu,std::process::id(),std::thread::current().id());file().unwrap().lock().unwrap().write_all(line.as_bytes()).expect("router trace write");}}}
