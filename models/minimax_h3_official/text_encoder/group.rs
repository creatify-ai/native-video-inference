//! Supervisor for a resident sequence-parallel conditioner group: one serving
//! process per GPU, every request delivered to all ranks. Rank 0 owns the reply
//! and the delivered files; peers write only their own scratch directories.
//! Identical input rejections keep the group. A GPU-phase failure, early rank
//! exit or watchdog expiry stops every rank and this supervisor.
use crate::Result;
use serde_json::{json, Value};
use std::{env, fs, io::{BufRead, BufReader, Read, Write}, os::unix::net::{UnixListener, UnixStream},
          path::{Path, PathBuf}, process::{Child, Command, Stdio}, sync::mpsc, time::{Duration, Instant}};

struct Ranks { children: Vec<Child>, sockets: Vec<PathBuf>, root: PathBuf }
impl Ranks {
    fn stop(&mut self) {
        for child in &mut self.children { let _ = child.kill(); }
        for child in &mut self.children { let _ = child.wait(); }
    }
    fn check_alive(&mut self) -> Result<()> {
        for (rank, child) in self.children.iter_mut().enumerate() {
            if let Some(status) = child.try_wait()? { return Err(format!("conditioner rank {rank} exited: {status}").into()); }
        }
        Ok(())
    }
}
impl Drop for Ranks { fn drop(&mut self) { self.stop(); let _ = fs::remove_dir_all(&self.root); } }

/// Peers receive their own fresh output directory beside rank 0's.
fn rank_args(args: &[String], rank: usize) -> Result<Vec<String>> {
    let index = match args.first().map(String::as_str) {
        Some("preload" | "prepare-offload" | "offload" | "restore") => return Ok(args.to_vec()),
        Some("infer") => 5,
        Some("infer-ref2va") => 4,
        _ => return Err("unsupported conditioner group request".into()),
    };
    let mut local = args.to_vec();
    if rank == 0 { return Ok(local); }
    let output = PathBuf::from(local.get(index).ok_or("missing conditioner output")?);
    let name = output.file_name().and_then(|n| n.to_str()).ok_or("invalid conditioner output")?;
    local[index] = output.with_file_name(format!("{name}.qwen-rank-{rank:02}")).to_str().ok_or("invalid output path")?.into();
    Ok(local)
}

fn request_rank(socket: &Path, args: &[String], timeout: Duration) -> std::result::Result<Value, String> {
    (|| -> Result<Value> {
        let mut stream = UnixStream::connect(socket)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(Duration::from_secs(30)))?;
        writeln!(stream, "{}", json!({"args": args}))?;
        stream.flush()?;
        let mut line = String::new();
        BufReader::new(stream).take(8 * 1024 * 1024 + 1).read_line(&mut line)?;
        if line.len() > 8 * 1024 * 1024 || !line.ends_with('\n') { return Err("invalid conditioner rank response".into()); }
        Ok(serde_json::from_str(&line)?)
    })().map_err(|e| e.to_string())
}

fn rejected(error: &str) -> Value {
    json!({"ok":false,"error":error,"details":{"component":"conditioner","recoverable":true,"gpu_execution_started":false}})
}

/// Ok: all ranks succeeded, or all rejected the input before any GPU work.
/// Err: the group is no longer usable.
fn distributed(ranks: &mut Ranks, args: Vec<String>, timeout: Duration) -> Result<Value> {
    let calls = match (0..ranks.sockets.len()).map(|rank| rank_args(&args, rank)).collect::<Result<Vec<_>>>() {
        Ok(calls) => calls,
        Err(error) => return Ok(rejected(&error.to_string())),
    };
    // A peer alone rejecting its output path would leave the others waiting in
    // the admission collective; reject for the whole group instead.
    if calls[1..].iter().flatten().any(|a| a.contains(".qwen-rank-") && Path::new(a).exists()) {
        return Ok(rejected("conditioner rank output already exists"));
    }
    let started = Instant::now();
    let (tx, rx) = mpsc::channel();
    let mut threads = vec![];
    for (rank, (socket, local)) in ranks.sockets.iter().zip(calls).enumerate() {
        let (socket, tx) = (socket.clone(), tx.clone());
        threads.push(std::thread::spawn(move || { let _ = tx.send((rank, request_rank(&socket, &local, timeout))); }));
    }
    drop(tx);
    let mut replies = vec![Value::Null; ranks.children.len()];
    let mut completed = 0;
    let collected = (|| -> Result<()> {
        while completed < replies.len() {
            ranks.check_alive()?;
            if started.elapsed() > timeout { return Err("conditioner group watchdog expired".into()); }
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok((rank, reply)) => {
                    let reply = reply.map_err(|e| format!("conditioner rank {rank}: {e}"))?;
                    // A rank that failed after GPU work started has left its peers mid-collective.
                    if reply["ok"] != true && reply["details"]["recoverable"] != true {
                        return Err(format!("conditioner rank {rank} failed: {}", reply["error"]).into());
                    }
                    replies[rank] = reply;
                    completed += 1;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => (),
                Err(mpsc::RecvTimeoutError::Disconnected) => return Err("conditioner rank channels closed early".into()),
            }
        }
        Ok(())
    })();
    if collected.is_err() { ranks.stop(); }
    for thread in threads { let _ = thread.join(); }
    collected?;
    let succeeded = replies.iter().filter(|r| r["ok"] == true).count();
    if succeeded != 0 && succeeded != replies.len() {
        ranks.stop();
        return Err("conditioner ranks disagreed on request admission".into());
    }
    let mut reply = replies[0].clone();
    if succeeded != 0 {
        reply["details"]["group"] = json!({"world": replies.len(), "all_ranks_completed": true,
            "request_wall_seconds": started.elapsed().as_secs_f64(),
            "ranks": replies.iter().map(|r| json!({"parallel": r["details"]["parallel"],
                "request_wall_seconds": r["details"]["request_wall_seconds"], "forward_seconds": r["details"]["forward_seconds"],
                "load_seconds": r["details"]["load_seconds"], "weight_transfer": r["details"]["weight_transfer"]})).collect::<Vec<_>>()});
    }
    Ok(reply)
}

pub fn serve(socket: &Path, world: usize) -> Result<()> {
    if ![2, 4, 8].contains(&world) { return Err("conditioner group world must be 2, 4 or 8".into()); }
    if socket.exists() { return Err("socket path already exists".into()); }
    let parent = socket.parent().ok_or("conditioner socket needs a parent directory")?;
    fs::create_dir_all(parent)?;
    let root = parent.join(format!("qwen-ranks-{}", std::process::id()));
    fs::create_dir(&root)?;
    let id = root.join("nccl.id");
    let mut ranks = Ranks { children: vec![], sockets: vec![], root: root.clone() };
    // NCCL bootstrap listens in this process: it lives as long as the ranks.
    crate::native::write_unique_id(&id)?;
    let binary = env::current_exe()?;
    // Rank r uses visible device first + r; the controller keeps a smaller group
    // on the highest devices, away from device 0's media allocations.
    let first: usize = env::var("H3_QWEN_FIRST_DEVICE").unwrap_or_else(|_| "0".into()).parse()?;
    for rank in 0..world {
        let rank_socket = root.join(format!("rank-{rank}.sock"));
        let mut command = Command::new(&binary);
        command.arg("serve").arg(&rank_socket).env("H3_QWEN_WORLD", world.to_string())
            .env("H3_QWEN_RANK", rank.to_string()).env("H3_QWEN_DEVICE", (first + rank).to_string())
            .env("H3_QWEN_UNIQUE_ID_FILE", &id).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::inherit());
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::process::CommandExt;
            unsafe extern "C" { fn prctl(option: i32, ...) -> i32; fn getppid() -> i32; }
            let parent_pid = std::process::id() as i32;
            unsafe {
                command.pre_exec(move || {
                    if prctl(1, 9, 0, 0, 0) != 0 { return Err(std::io::Error::last_os_error()); }
                    if getppid() != parent_pid { return Err(std::io::Error::other("conditioner group parent exited during spawn")); }
                    Ok(())
                });
            }
        }
        ranks.children.push(command.spawn()?);
        ranks.sockets.push(rank_socket);
    }
    let started = Instant::now();
    while ranks.sockets.iter().any(|p| !p.exists()) {
        ranks.check_alive()?;
        if started.elapsed() > Duration::from_secs(300) { return Err("conditioner rank startup timeout".into()); }
        std::thread::sleep(Duration::from_millis(25));
    }
    let listener = UnixListener::bind(socket)?;
    let timeout = Duration::from_millis(env::var("H3_QWEN_REQUEST_TIMEOUT_MS").unwrap_or_else(|_| "1800000".into()).parse()?);
    eprintln!("dynamic conditioner group ready {} world={world}", socket.display());
    for connection in listener.incoming() {
        let mut stream = connection?;
        let result = (|| -> Result<Value> {
            stream.set_read_timeout(Some(Duration::from_secs(30)))?;
            let mut line = String::new();
            BufReader::new(&stream).take((1 << 20) + 1).read_line(&mut line)?;
            if line.len() > 1 << 20 || !line.ends_with('\n') { return Err("invalid newline-delimited request".into()); }
            let request: Value = serde_json::from_str(&line)?;
            let args: Vec<String> = serde_json::from_value(request["args"].clone())?;
            distributed(&mut ranks, args, timeout)
        })();
        match result {
            Ok(reply) => { writeln!(stream, "{}", serde_json::to_string(&reply)?)?; stream.flush()?; }
            Err(error) => {
                ranks.stop();
                let _ = writeln!(stream, "{}", json!({"ok":false,"error":error.to_string(),
                    "details":{"component":"conditioner","recoverable":false,"gpu_execution_started":true}}));
                return Err(error);
            }
        }
    }
    Ok(())
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn weight_transfers_reach_every_rank_without_output_rewriting()->Result<()> {
        for command in ["prepare-offload","offload","restore"] {
            let args=vec![command.to_string()];
            for rank in 0..8 {assert_eq!(rank_args(&args,rank)?,args);}
        }
        assert!(rank_args(&["unknown".into()],0).is_err());Ok(())
    }
}
