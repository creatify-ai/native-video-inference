//! Request facts and checked component use over the framework startup executor.
use crate::Result;
use serde_json::{json, Value};
use std::{io::{BufRead, BufReader, Read, Write}, os::unix::net::UnixStream, path::Path, time::Duration};
const KEY: &str = "NATIVE_STARTUP_GATE";
const SCHEMA: &str = "native-startup-gate-v1";
const LIMIT: usize = 65536;

pub fn enabled() -> bool { std::env::var_os(KEY).is_some() }

fn exchange(mut message: Value) -> Result<()> {
    let raw = std::env::var(KEY)?;
    if raw.len() > 1024 { return Err("startup gate pin exceeds its bound".into()); }
    let pin: Value = serde_json::from_str(&raw)?;
    let path = pin["path"].as_str().ok_or("startup gate path missing")?;
    let token = pin["token"].as_str().ok_or("startup gate token missing")?;
    if pin["schema"] != SCHEMA || !Path::new(path).is_absolute()
        || Path::new(path).components().any(|c| matches!(c, std::path::Component::ParentDir))
        || token.len() != 32 || !token.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("invalid framework startup gate identity".into());
    }
    message["schema"] = json!(SCHEMA); message["token"] = json!(token);
    let mut encoded = serde_json::to_vec(&message)?; encoded.push(b'\n');
    if encoded.len() > LIMIT { return Err("startup gate frame exceeds its bound".into()); }
    let mut stream = UnixStream::connect(path)?;
    stream.set_read_timeout(Some(Duration::from_secs(900)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    stream.write_all(&encoded)?;
    let mut reply = Vec::new();
    BufReader::new(stream).take((LIMIT+1) as u64).read_until(b'\n', &mut reply)?;
    if reply.len() > LIMIT || !reply.ends_with(b"\n") { return Err("invalid bounded startup acknowledgement".into()); }
    let reply: Value = serde_json::from_slice(&reply)?;
    if reply != json!({"schema":SCHEMA,"token":token,"kind":message["kind"],"scope":message["scope"],"continued":true}) {
        return Err(format!("framework rejected component startup: {reply}").into());
    }
    Ok(())
}

pub fn inputs(scope: &Path, values: Value) -> Result<()> {
    if enabled() { exchange(json!({"kind":"inputs","scope":scope,"inputs":values}))?; }
    Ok(())
}

pub fn before_use(component: &str) -> Result<()> {
    before_use_command(component, None)
}

/// The transport can name a command when several logical consumers share an
/// actor. The model's admitted adapter resolves it to the same startup gate.
pub fn before_use_command(component: &str, command: Option<&str>) -> Result<()> {
    if enabled() {
        let scope = std::env::var("NATIVE_STARTUP_SCOPE")?;
        let mut message=json!({"kind":"before_use","scope":scope,"component":component});
        if let Some(command)=command { message["command"]=json!(command); }
        exchange(message)?;
    }
    Ok(())
}

/// Check a declared consumer for a caller-owned request scope.
/// Used by native orchestrators whose child environment holds the same scope.
pub fn before_use_in_scope(scope:&Path, component:&str)->Result<()> {
    before_use_command_in_scope(scope, component, None)
}

/// Check a declared consumer for a caller-owned scope, naming the command its own
/// transport sends: an adapter that resolves shared actors by command requires it.
pub fn before_use_command_in_scope(scope:&Path, component:&str, command:Option<&str>)->Result<()> {
    if enabled() {
        let mut message=json!({"kind":"before_use","scope":scope,"component":component});
        if let Some(command)=command { message["command"]=json!(command); }
        exchange(message)?;
    }
    Ok(())
}

#[cfg(test)] mod tests {
    use super::*;
    use std::{os::unix::net::UnixListener, sync::atomic::{AtomicUsize, Ordering}, thread};
    #[test] fn transport_requires_its_matching_framework_acknowledgement() {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        for accepted in [true, false] {
            let path=std::env::temp_dir().join(format!("startup-{}-{}.sock",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
            let listener=UnixListener::bind(&path).unwrap();
            let worker=thread::spawn(move||(0..2).map(|_|{
                let (mut peer,_)=listener.accept().unwrap();
                let mut raw=String::new();BufReader::new(peer.try_clone().unwrap()).read_line(&mut raw).unwrap();
                let message:Value=serde_json::from_str(&raw).unwrap();
                assert_eq!(message["kind"],"before_use");assert_eq!(message["component"],"dit");
                let reply=json!({"schema":SCHEMA,"token":message["token"],"kind":message["kind"],
                    "scope":message["scope"],"continued":accepted});
                writeln!(peer,"{reply}").unwrap();
                message
            }).collect::<Vec<_>>());
            std::env::set_var(KEY,json!({"schema":SCHEMA,"path":path,"token":"a".repeat(32)}).to_string());
            std::env::set_var("NATIVE_STARTUP_SCOPE","/request/output");
            let result=before_use("dit");
            let scoped=before_use_command_in_scope(Path::new("/request/base"),"dit",Some("/models/dit"));
            std::env::remove_var(KEY);std::env::remove_var("NATIVE_STARTUP_SCOPE");
            let messages=worker.join().unwrap();std::fs::remove_file(path).unwrap();
            assert_eq!((result.is_ok(),scoped.is_ok()),(accepted,accepted));
            assert_eq!((&messages[0]["scope"],messages[0].get("command")),(&json!("/request/output"),None));
            assert_eq!((&messages[1]["scope"],&messages[1]["command"]),(&json!("/request/base"),&json!("/models/dit")));
        }
    }
}
