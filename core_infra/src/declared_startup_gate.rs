//! Native request-fact and consumer transport for the complete executor.
//! Readiness is checked in Rust and wakes on executor events, not Python polling.
use crate::declared_startup::{DeclaredStartup, Result, Work};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::Shutdown;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const SCHEMA: &str = "native-startup-gate-v1";
const MAX_FRAME: u64 = 65536;
struct State {
    peers: BTreeMap<u64, UnixStream>, consumers: BTreeSet<(String, String)>,
    workers: Vec<JoinHandle<()>>, error: Option<String>,
}
pub struct StartupGate {
    path: PathBuf, token: String, state: Arc<Mutex<State>>, closed: Arc<AtomicBool>,
    accept: Option<JoinHandle<()>>, executor: Arc<DeclaredStartup>,
}
impl StartupGate {
    pub fn open(executor: Arc<DeclaredStartup>, path: &Path, token: &str, publish: Work) -> Result<Self> {
        if !path.is_absolute() || path.as_os_str().len() >= 104 || token.len() != 32
            || !token.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
            return Err("startup gate needs a short absolute path and identity".into());
        }
        let listener = UnixListener::bind(path).map_err(|e| format!("startup gate bind: {e}"))?;
        let state = Arc::new(Mutex::new(State { peers: BTreeMap::new(), consumers: BTreeSet::new(), workers: vec![], error: None }));
        let closed = Arc::new(AtomicBool::new(false));
        let (shared, stop, runtime, identity) = (state.clone(), closed.clone(), executor.clone(), token.to_string());
        let accept = thread::Builder::new().name("startup-gate".into()).spawn(move || {
            let ids = AtomicU64::new(1);
            for connection in listener.incoming() {
                if stop.load(Ordering::Acquire) { break; }
                let stream = match connection { Ok(v) => v, Err(e) => { shared.lock().unwrap().error = Some(e.to_string()); break; } };
                let id = ids.fetch_add(1, Ordering::Relaxed);
                {
                    let mut state = shared.lock().unwrap();
                    if stop.load(Ordering::Acquire) { break; }
                    if state.peers.len() >= 16 { state.error = Some("startup gate peer limit exceeded".into()); continue; }
                    match stream.try_clone() { Ok(v) => { state.peers.insert(id, v); }, Err(e) => { state.error = Some(e.to_string()); continue; } }
                }
                let (state, closed, executor, token, publish) = (shared.clone(), stop.clone(), runtime.clone(), identity.clone(), publish.clone());
                match thread::Builder::new().name("startup-gate-peer".into()).spawn(move || {
                    let outcome = Self::serve(&stream, &token, &executor, &closed, &publish, &state);
                    let mut state = state.lock().unwrap();
                    state.peers.remove(&id);
                    if !closed.load(Ordering::Acquire) { if let Err(error) = outcome { state.error.get_or_insert(error); } }
                }) {
                    Ok(worker) => shared.lock().unwrap().workers.push(worker),
                    Err(e) => { let mut state = shared.lock().unwrap(); state.peers.remove(&id); state.error = Some(e.to_string()); }
                }
            }
        }).map_err(|e| { let _ = std::fs::remove_file(path); format!("startup gate thread: {e}") })?;
        Ok(Self { path: path.into(), token: token.into(), state, closed, accept: Some(accept), executor })
    }
    fn serve(mut stream: &UnixStream, token: &str, executor: &DeclaredStartup, closed: &AtomicBool,
            publish: &Work, state: &Mutex<State>) -> Result<()> {
        stream.set_read_timeout(Some(Duration::from_secs(900))).map_err(|e| e.to_string())?;
        stream.set_write_timeout(Some(Duration::from_secs(30))).map_err(|e| e.to_string())?;
        let mut raw = vec![];
        BufReader::new(stream).take(MAX_FRAME + 1).read_until(b'\n', &mut raw).map_err(|e| e.to_string())?;
        if raw.len() as u64 > MAX_FRAME || !raw.ends_with(b"\n") { return Err("invalid bounded startup gate frame".into()); }
        let message: Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
        let scope = message["scope"].as_str().filter(|s| !s.is_empty()).ok_or("startup gate scope missing")?;
        let kind = message["kind"].as_str().ok_or("startup gate kind missing")?;
        let outcome: Result<()> = (|| {
            if message["schema"] != SCHEMA || message["token"] != token { return Err("startup gate frame differs from its admitted request".into()); }
            match kind {
                "inputs" => {
                    publish(&json!({"scope":scope,"inputs":message["inputs"]}))?;
                    if executor.snapshot()["scope"] != scope { return Err("native inputs did not publish this request scope".into()); }
                },
                "before_use" => {
                    let component = message["component"].as_str().ok_or("startup gate component missing")?;
                    executor.before_use_interruptible(Some(scope), component, Some(closed))?;
                    state.lock().unwrap().consumers.insert((scope.into(), component.into()));
                },
                _ => return Err("invalid startup gate message kind".into()),
            }
            Ok(())
        })();
        let mut reply = json!({"schema":SCHEMA,"token":token,"kind":kind,"scope":scope,"continued":outcome.is_ok()});
        if let Err(error) = &outcome {
            reply["error"] = json!(error);
            // Record the failure before the peer sees the reply, so a poll that follows the reply reports it.
            if !closed.load(Ordering::Acquire) { state.lock().unwrap().error.get_or_insert_with(|| error.clone()); }
        }
        let mut bytes = serde_json::to_vec(&reply).map_err(|e| e.to_string())?; bytes.push(b'\n');
        stream.write_all(&bytes).map_err(|e| e.to_string())?;
        outcome
    }
    pub fn environment(&self) -> Value { json!({"NATIVE_STARTUP_GATE":json!({"schema":SCHEMA,"path":self.path,"token":self.token}).to_string()}) }
    pub fn status(&self) -> Result<Value> {
        let state = self.state.lock().unwrap();
        if let Some(error) = &state.error { return Err(error.clone()); }
        Ok(json!({"consumers":state.consumers,"closed":self.closed.load(Ordering::Acquire),"executor":"rust"}))
    }
    pub fn poll(&self) -> Result<()> {
        self.executor.poll()?;
        let state=self.state.lock().unwrap();
        if let Some(error)=&state.error {return Err(error.clone());}
        Ok(())
    }
    pub fn close(&mut self) -> Result<()> {
        if self.closed.swap(true, Ordering::AcqRel) { return Ok(()); }
        // Shut down blocked readers and wake readiness waiters before joining.
        for stream in self.state.lock().unwrap().peers.values() { let _ = stream.shutdown(Shutdown::Both); }
        self.executor.wake();
        let _ = UnixStream::connect(&self.path);
        if let Some(accept) = self.accept.take() { accept.join().map_err(|_| "startup gate accept thread panicked")?; }
        let workers = std::mem::take(&mut self.state.lock().unwrap().workers);
        for worker in workers { worker.join().map_err(|_| "startup gate peer panicked")?; }
        std::fs::remove_file(&self.path).map_err(|e| e.to_string())?;
        Ok(())
    }
}
impl Drop for StartupGate { fn drop(&mut self) { let _ = self.close(); } }
