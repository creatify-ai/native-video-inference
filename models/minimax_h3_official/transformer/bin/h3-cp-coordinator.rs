//! Native, serialized-request supervisor for resident context-parallel ranks.
//! One rank error poisons the group: stop all children, never return rank-0-only success.
pub use h3_official_native_run::startup_trace;
use serde_json::{json,Value};
use std::{env,fs,io::{BufRead,BufReader,Read,Write},os::unix::net::{UnixListener,UnixStream},
    path::{Path,PathBuf},process::{Child,Command,Stdio},sync::mpsc,time::{Duration,Instant}};
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
struct Ranks {children:Vec<Child>,sockets:Vec<PathBuf>,id:PathBuf}
impl Ranks {
    fn stop(&mut self){for child in &mut self.children{let _=child.kill();}for child in &mut self.children{let _=child.wait();}}
    fn check_alive(&mut self)->Result<()> {
        for (rank,child) in self.children.iter_mut().enumerate(){
            if let Some(status)=child.try_wait()?{return Err(format!("CP rank {rank} exited: {status}").into());}
        }Ok(())
    }
}
impl Drop for Ranks {
    fn drop(&mut self){self.stop();for path in &self.sockets{let _=fs::remove_file(path);}let _=fs::remove_file(&self.id);}
}
fn request_rank(socket:PathBuf,args:Vec<String>,timeout:Duration)->std::result::Result<Value,String> {
    (||->Result<Value>{
        let mut stream=UnixStream::connect(socket)?;
        stream.set_read_timeout(Some(timeout))?;stream.set_write_timeout(Some(Duration::from_secs(30)))?;
        writeln!(stream,"{}",json!({"args":args}))?;stream.flush()?;
        let mut line=String::new();BufReader::new(stream).take(8*1024*1024+1).read_line(&mut line)?;
        if line.len()>8*1024*1024||!line.ends_with('\n'){return Err("invalid CP rank response".into());}
        let value:Value=serde_json::from_str(&line)?;
        if value["ok"]!=true{return Err(format!("rank request failed: {}",value["error"]).into());}
        Ok(value)
    })().map_err(|e|e.to_string())
}
/// One call per rank, in parallel; every rank must answer ok. A failure stops the group.
fn fan_out(ranks:&mut Ranks,calls:Vec<Vec<String>>,timeout:Duration)->Result<Vec<Value>> {
    let started=Instant::now();let(tx,rx)=mpsc::channel();let mut threads=Vec::new();
    for(rank,(socket,local))in ranks.sockets.iter().zip(calls).enumerate(){
        let socket=socket.clone();let tx=tx.clone();
        threads.push(std::thread::spawn(move||{let _=tx.send((rank,request_rank(socket,local,timeout)));}));
    }
    drop(tx);let mut replies=vec![Value::Null;ranks.children.len()];let mut completed=0;
    let result=(||->Result<()>{
        while completed<replies.len(){
            ranks.check_alive()?;
            if started.elapsed()>timeout{return Err("CP distributed request watchdog expired".into());}
            match rx.recv_timeout(Duration::from_millis(100)){
                Ok((rank,reply))=>{if !replies[rank].is_null(){return Err("duplicate CP rank reply".into());}
                    replies[rank]=reply.map_err(|e|format!("rank {rank}: {e}"))?;completed+=1;},
                Err(mpsc::RecvTimeoutError::Timeout)=>(),
                Err(mpsc::RecvTimeoutError::Disconnected)=>return Err("CP response channels closed early".into()),
            }
        }Ok(())
    })();
    if result.is_err(){ranks.stop();}
    for thread in threads{let _=thread.join();}
    result?;Ok(replies)
}
fn distributed_request(ranks:&mut Ranks,args:Vec<String>,timeout:Duration)->Result<Value> {
    if args.len()==4&&args[0]=="preload" {
        // Every rank loads its own shards; rank 0 answers for the group.
        let world=ranks.sockets.len();
        let replies=fan_out(ranks,vec![args;world],timeout)?;
        let mut response=replies[0].clone();
        response["details"]["context_parallel"]=json!({"world":world,"all_ranks_completed":true,
            "load_seconds":replies.iter().map(|r|r["details"]["load_seconds"].clone()).collect::<Vec<_>>()});
        return Ok(response);
    }
    if args.len()==1&&args[0]=="verify-gpu-weights" {
        let world=ranks.sockets.len();
        let replies=fan_out(ranks,vec![args;world],timeout)?;
        return Ok(json!({"ok":true,"details":{"component":"dit","world_size":world,
            "all_ranks_completed":true,"ranks":replies.iter().map(|r|r["details"].clone()).collect::<Vec<_>>()}}));
    }
    if args.len()==1&&args[0]=="await-release" {
        // Each rank answers only after the weight-cache release that followed its last reply.
        let world=ranks.sockets.len();
        let replies=fan_out(ranks,vec![args;world],timeout)?;
        let mut response=replies[0].clone();
        response["details"]["context_parallel"]=json!({"world":replies.len(),
            "release_seconds":replies.iter().map(|r|r["details"]["release_seconds"].clone()).collect::<Vec<_>>()});
        return Ok(response);
    }
    // prepare-layout MODEL ROUTER DENSE OUTPUT NFE REQUEST: same fan-out, no report.
    let prepare=args.first().is_some_and(|a|a=="prepare-layout");
    let out_index=if prepare {
        if args.len()!=7||!args[5].parse::<usize>().is_ok_and(|n|(1..=999).contains(&n)){return Err("expected prepare-layout MODEL ROUTER DENSE OUTPUT NFE REQUEST".into());}
        4
    }else{
        if args.len()!=7||args[4]!="generate"||!args[5].parse::<usize>().is_ok_and(|n|(1..=999).contains(&n)){return Err("expected MODEL ROUTER DENSE OUTPUT generate NFE REQUEST".into());}
        3
    };
    let output=PathBuf::from(&args[out_index]);
    if !output.is_absolute()||output.exists(){return Err("CP output must be a fresh absolute directory".into());}
    let mut calls=Vec::new();
    for rank in 0..ranks.sockets.len(){
        let mut local=args.clone();
        if rank>0 {
            let name=output.file_name().ok_or("missing output basename")?.to_str().ok_or("invalid output basename")?;
            local[out_index]=output.with_file_name(format!("{name}.cp-rank-{rank:02}")).to_str().ok_or("invalid output path")?.into();
            if Path::new(&local[out_index]).exists(){return Err("CP peer output already exists".into());}
        }
        calls.push(local);
    }
    let started=Instant::now();
    let replies=fan_out(ranks,calls,timeout)?;
    let mut response=replies[0].clone();
    if prepare {
        response["details"]["context_parallel"]=json!({"world":replies.len(),"all_ranks_completed":true,
            "prepared":replies.iter().map(|r|r["details"]["prepared"].clone()).collect::<Vec<_>>()});
        return Ok(response);
    }
    let request_wall=started.elapsed().as_secs_f64();let post=Instant::now();
    // One timing summary per rank (each rank's report.json keeps its full details) and a compact
    // rewrite. H3_CP_COMPACT_SUPERVISOR=0 copies every rank's full details into rank 0's report.
    let compact=env::var("H3_CP_COMPACT_SUPERVISOR").as_deref()!=Ok("0");
    let rank_details:Vec<Value>=if compact {
        replies.iter().enumerate().map(|(rank,r)|{let d=&r["details"];json!({"rank":rank,"requests_completed":d["requests_completed"],
            "request_wall_seconds":d["request_wall_seconds"],"preflight_seconds":d["preflight_seconds"],"execute_seconds":d["execute_seconds"],
            "finalize_seconds":d["finalize_seconds"],"report_write_seconds":d["report_write_seconds"],"after_wall_seconds":d["after_wall_seconds"],
            "details":format!("{}/report.json",if rank==0{output.display().to_string()}else{format!("{}.cp-rank-{rank:02}",output.display())})})}).collect()
    }else{replies.iter().map(|r|r["details"].clone()).collect()};
    response["details"]["context_parallel"]=json!({"world":replies.len(),"all_ranks_completed":true,
        "request_wall_seconds":request_wall,"rank_details":rank_details});
    // Separate protocol completion from independent numerical acceptance.
    let mut report:Value=serde_json::from_slice(&crate::startup_trace::read(output.join("report.json"))?)?;
    report["context_parallel_supervisor"]=response["details"]["context_parallel"].clone();
    fs::write(output.join("report.json"),if compact{serde_json::to_vec(&report)?}else{serde_json::to_vec_pretty(&report)?})?;
    response["details"]["context_parallel"]["post_seconds"]=json!(post.elapsed().as_secs_f64());
    Ok(response)
}
fn run()->Result<()> {
    let args:Vec<String>=env::args().collect();
    // `serve SOCKET` is the shared owner command line; the rank program and world then come from
    // H3_CP_RANK_PROGRAM and H3_CP_WORLD.
    let (socket,binary,world)=match args.as_slice() {
        [_,serve,socket,binary,world] if serve=="serve"=>(PathBuf::from(socket),PathBuf::from(binary),world.parse::<usize>()?),
        [_,serve,socket] if serve=="serve"=>(PathBuf::from(socket),
            PathBuf::from(env::var("H3_CP_RANK_PROGRAM").map_err(|_|"serve SOCKET requires H3_CP_RANK_PROGRAM")?),
            env::var("H3_CP_WORLD").map_err(|_|"serve SOCKET requires H3_CP_WORLD")?.parse::<usize>()?),
        _=>return Err("usage: h3-cp-coordinator serve SOCKET [RANK_BINARY WORLD]".into()),
    };
    if ![1,2,4,8].contains(&world)||!binary.is_absolute()||!binary.is_file(){return Err("invalid CP world/binary".into());}
    if socket.exists(){return Err("coordinator socket exists".into());}
    let parent=socket.parent().ok_or("coordinator socket needs parent")?;fs::create_dir_all(parent)?;
    let root=parent.join(format!("cp-ranks-{}",std::process::id()));fs::create_dir(&root)?;
    let id=root.join("nccl.id");
    // Rank logs stay beside the coordinator's own when H3_CP_RANK_LOG_DIR names a directory.
    let logs=env::var_os("H3_CP_RANK_LOG_DIR").map(PathBuf::from).unwrap_or_else(||root.clone());
    let mut ranks=Ranks{children:vec![],sockets:vec![],id:id.clone()};
    if world>1 {
        // NCCL bootstrap may own a listener in the ID creator. Keep that
        // creator alive for all resident rank initialization/reinitialization.
        h3_official_native_run::native::cp_write_unique_id(&id)?;
    }
    for rank in 0..world {
        let rank_socket=root.join(format!("rank-{rank}.sock"));
        let log=fs::File::create(logs.join(format!("rank-{rank}.log")))?;
        let mut command=Command::new(&binary);
        command.arg("serve").arg(&rank_socket).env("H3_CP_WORLD",world.to_string()).env("H3_CP_RANK",rank.to_string())
            .env("H3_CP_DEVICE",rank.to_string()).env("H3_CP_UNIQUE_ID_FILE",&id)
            .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::from(log));
        #[cfg(target_os="linux")]
        {
            use std::os::unix::process::CommandExt;
            unsafe extern "C" {fn prctl(option:i32,...)->i32;fn getppid()->i32;}
            let parent_pid=std::process::id() as i32;
            unsafe{command.pre_exec(move||{
                if prctl(1,9,0,0,0)!=0{return Err(std::io::Error::last_os_error());}
                if getppid()!=parent_pid{return Err(std::io::Error::other("CP parent exited during spawn"));}
                Ok(())
            });}
        }
        let child=command.spawn()?;
        ranks.children.push(child);ranks.sockets.push(rank_socket);
    }
    let started=Instant::now();
    while ranks.sockets.iter().any(|p|!p.exists()){
        ranks.check_alive()?;if started.elapsed()>Duration::from_secs(60){return Err("CP rank startup timeout".into());}
        std::thread::sleep(Duration::from_millis(25));
    }
    let listener=UnixListener::bind(&socket)?;
    let timeout=Duration::from_millis(env::var("H3_CP_REQUEST_TIMEOUT_MS").unwrap_or_else(|_|"1800000".into()).parse()?);
    eprintln!("native CP coordinator ready: world={world}");
    for connection in listener.incoming(){
        let mut stream=connection?;
        let result=(||->Result<Value>{
            stream.set_read_timeout(Some(Duration::from_secs(30)))?;
            let mut line=String::new();BufReader::new(&stream).take(1048577).read_line(&mut line)?;
            if line.len()>1<<20||!line.ends_with('\n'){return Err("invalid CP coordinator request".into());}
            let request:Value=serde_json::from_str(&line)?;
            let call:Vec<String>=serde_json::from_value(request["args"].clone())?;
            distributed_request(&mut ranks,call,timeout)
        })();
        match result {
            Ok(reply)=>{writeln!(stream,"{reply}")?;stream.flush()?;},
            Err(error)=>{
                ranks.stop();
                let _=writeln!(stream,"{}",json!({"ok":false,"error":error.to_string(),"details":{"component":"dit","recoverable":false}}));
                return Err(error);
            }
        }
    }Ok(())
}
fn main(){if let Err(error)=run(){eprintln!("{error}");std::process::exit(1);}}
