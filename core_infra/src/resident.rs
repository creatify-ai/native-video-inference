//! The resident native process that core_infra/runtime owns (`ProcessGroup.owner` / `NativeOwner`
//! spawn `<program> serve <socket>`). It binds its Unix socket before any CUDA work, so its owner
//! can connect at once, then serves one connection at a time: one JSON line `{"args": [...]}` in,
//! one line `{"ok": bool, "error": ..., "details": {...}}` out. The handler owns all state, e.g.
//! models loaded by the first request that needs them and kept for later ones; `shutdown` ends the
//! loop after replying.
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixListener;
use std::path::Path;

/// Request lines are bounded like the Python client's (64 KiB).
const MAX_REQUEST: u64 = 65536;

pub trait Handler {
    /// One command: `Ok(details)` or `Err(message)`.
    fn handle(&mut self, args: &[String]) -> Result<Value, String>;
}

fn reply(handler: &mut dyn Handler, line: &str) -> (Value, bool) {
    let args: Vec<String> = match serde_json::from_str::<Value>(line).ok().and_then(|v| {
        v["args"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
    }) {
        Some(args) => args,
        None => return (json!({"ok": false, "error": "request must be one JSON line {\"args\": [strings]}"}), false),
    };
    if args.first().map(String::as_str) == Some("shutdown") {
        return (json!({"ok": true, "details": {"shutdown": true}}), true);
    }
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler.handle(&args)));
    (match outcome {
        Ok(Ok(details)) => json!({"ok": true, "details": details}),
        Ok(Err(error)) => json!({"ok": false, "error": error}),
        Err(_) => json!({"ok": false, "error": "native handler panicked"}),
    }, false)
}

/// Binds `socket` (replacing a stale file) and serves until `shutdown`.
pub fn serve(socket: &Path, handler: &mut dyn Handler) -> std::io::Result<()> {
    if socket.exists() { std::fs::remove_file(socket)?; }
    let listener = UnixListener::bind(socket)?;
    for stream in listener.incoming() {
        let mut stream = stream?;
        let mut line = String::new();
        BufReader::new((&stream).take(MAX_REQUEST + 1)).read_line(&mut line)?;
        let (out, stop) = if line.len() as u64 > MAX_REQUEST || !line.ends_with('\n') {
            (json!({"ok": false, "error": "request exceeds the transport bound"}), false)
        } else {
            reply(handler, &line)
        };
        let mut bytes = serde_json::to_vec(&out).map_err(std::io::Error::other)?;
        bytes.push(b'\n');
        let _ = stream.write_all(&bytes);
        if stop { break; }
    }
    let _ = std::fs::remove_file(socket);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Echo(usize);
    impl Handler for Echo {
        fn handle(&mut self, args: &[String]) -> Result<Value, String> {
            self.0 += 1;
            match args.first().map(String::as_str) {
                Some("fail") => Err("injected".into()),
                Some("panic") => panic!("boom"),
                _ => Ok(json!({"args": args, "count": self.0})),
            }
        }
    }
    fn call(socket: &Path, args: &[&str]) -> Value {
        let mut stream = std::os::unix::net::UnixStream::connect(socket).unwrap();
        stream.write_all(format!("{}\n", json!({"args": args})).as_bytes()).unwrap();
        let mut line = String::new();
        BufReader::new(&stream).read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
    }
    #[test] fn serves_commands_errors_panics_and_shutdown() {
        let socket = std::env::temp_dir().join(format!("native-resident-{}.sock", std::process::id()));
        let path = socket.clone();
        let server = std::thread::spawn(move || serve(&path, &mut Echo(0)).unwrap());
        while !socket.exists() { std::thread::sleep(std::time::Duration::from_millis(5)); }
        let ok = call(&socket, &["load", "x"]);
        assert_eq!((ok["ok"].as_bool(), ok["details"]["count"].as_u64()), (Some(true), Some(1)));
        assert_eq!(call(&socket, &["fail"])["error"], "injected");
        assert_eq!(call(&socket, &["panic"])["error"], "native handler panicked");
        assert_eq!(call(&socket, &["again"])["details"]["count"], 4);
        assert_eq!(call(&socket, &["shutdown"])["details"]["shutdown"], true);
        server.join().unwrap();
        assert!(!socket.exists());
    }
}
