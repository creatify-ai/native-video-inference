//! One public checkpoint owner, with replaceable request layouts.
use crate::{native::{PreparedRequest, ResidentModel, Runtime}, profile, Result};
use serde_json::{json, Value};
use std::{env, fs, io::{BufRead, BufReader, Read, Write}, os::unix::net::UnixListener,
    path::{Path, PathBuf}, rc::Rc, time::Instant};

#[derive(Default)]
struct Server {
    model:Option<Rc<ResidentModel>>, runtime:Option<Runtime>, paths:Option<Vec<String>>,
    identity:String, requests:usize, layouts:usize, initial_load:f64, poisoned:bool,
}
impl Server {
    fn load(&mut self, paths:&[String])->Result<Value> {
        if paths.len()!=3 {return Err("preload requires MODEL ROUTER DENSE".into());}
        if let Some(current)=&self.paths {
            if current!=paths {return Err("public owner is already bound to a different checkpoint".into());}
        }
        if self.model.is_none() {
            self.poisoned=true;
            crate::pointwise_opt::begin()?;crate::bsa_pack::begin()?;
            let model=ResidentModel::preload(Path::new(&paths[0]),Path::new(&paths[2]),None,0)?;
            self.initial_load=model.load_seconds();self.identity=model.identity()?;
            self.model=Some(model);self.paths=Some(paths.to_vec());self.poisoned=false;
        }
        Ok(json!({"component":"dit","model_load_count":1,"requests_completed":self.requests,
            "load_seconds":self.initial_load,"ownership":self.model.as_ref().unwrap().ownership_report()?}))
    }
    fn build_layout(&mut self,paths:&[String],output:&Path,input:&Path,prepared:&PreparedRequest)->Result<bool> {
        self.load(paths)?;
        if let Some(runtime)=&self.runtime {if runtime.prepared_fits(prepared)? {return Ok(false);}}
        self.poisoned=true;
        self.runtime=None;
        self.runtime=Some(Runtime::new_shared(Path::new(&paths[0]),Path::new(&paths[1]),
            Path::new(&paths[2]),output,input,self.model.clone())?);
        self.layouts+=1;self.poisoned=false;Ok(true)
    }
    fn request(&mut self,args:&[String])->Result<Value> {
        match args.first().map(String::as_str) {
            Some("preload") if args.len()==4 => return self.load(&args[1..]),
            Some("await-release") if args.len()==1 => return Ok(json!({"component":"dit","released":true})),
            Some("verify-gpu-weights") if args.len()==1 => {
                return self.model.as_ref().ok_or("no public model is resident")?.verify_loaded_payload();
            },
            Some("prepare-layout") => {
                if args.len()!=7 {return Err("expected prepare-layout MODEL ROUTER DENSE OUTPUT NFE REQUEST".into());}
                let prepared=PreparedRequest::read(Path::new(&args[6]))?;
                if args[5].parse::<usize>()?!=prepared.nfe()? {return Err("prepared NFE differs from request".into());}
                if prepared.is_windowed() || prepared.transaction()?.is_some() {
                    return Err("public entry accepts complete official-partition requests".into());
                }
                let output=Path::new(&args[4]);
                if !output.is_absolute() || output.exists() {return Err("layout output must be a fresh absolute path".into());}
                let rebuilt=self.build_layout(&args[1..4],output,prepared.input_root(),&prepared)?;
                return Ok(json!({"component":"dit","prepared":true,"layout_rebuilt":rebuilt,"forward_executed":false}));
            },
            _=>(),
        }
        if args.len()!=7 || args[4]!="generate" {return Err("expected MODEL ROUTER DENSE OUTPUT generate NFE REQUEST".into());}
        let input=Path::new(&args[6]);let output=Path::new(&args[3]);
        if !output.is_absolute() || output.exists() {return Err("output must be a fresh absolute path".into());}
        let prepared=PreparedRequest::read(input)?;
        let nfe=prepared.nfe()?;
        if args[5].parse::<usize>()?!=nfe || prepared.structure_only() || prepared.is_windowed() || prepared.transaction()?.is_some() {
            return Err("public generation requires real inputs and a complete official schedule".into());
        }
        let started=Instant::now();
        crate::debug_audit::begin()?;crate::pointwise_opt::begin()?;crate::bsa_pack::begin()?;
        profile::begin_request(self.requests)?;
        let rebuilt=self.build_layout(&args[..3],output,input,&prepared)?;
        self.poisoned=true;
        let runtime=self.runtime.as_mut().ok_or("public request layout missing")?;
        runtime.reset_prepared(prepared,output)?;
        if runtime.nfe()!=nfe {return Err("request NFE differs from resident schedule".into());}
        let first_dit=native_core_infra::execution_start::Capture::begin();
        let mut report=runtime.execute("generate",nfe)?;
        if let Some(marker)=first_dit.report()?{report["first_dit_start"]=marker;}
        if report["passed"]!=true {return Err("public native execution failed its output gate".into());}
        let packing=crate::bsa_pack::finish(runtime.sparse_calls_expected()?)?;
        if runtime.model_identity()?!=self.identity {return Err("resident model identity changed during generation".into());}
        self.requests+=1;
        let details=json!({"component":"dit","model_load_count":1,"requests_completed":self.requests,
            "initial_model_load_seconds":self.initial_load,"layout_build_count":self.layouts,
            "layout_rebuilt":rebuilt,"request_wall_seconds":started.elapsed().as_secs_f64(),
            "weights_identity_sha256":self.identity,"bsa_pack":packing,
            "sparse_attention":runtime.sparse_attention_report()?,"profile":profile::finish()?,
            "numerical_acceptance":"external SGLang latent comparison required"});
        report["resident"]=details.clone();report["context_parallel"]=runtime.cp_report();
        fs::write(output.join("report.json"),serde_json::to_vec_pretty(&report)?)?;
        runtime.release_deferred_weight_cache()?;
        self.poisoned=false;Ok(details)
    }
}
struct SocketGuard(PathBuf);
impl Drop for SocketGuard {fn drop(&mut self){let _=fs::remove_file(&self.0);}}
fn run()->Result<()> {
    let args:Vec<String>=env::args().collect();
    if args.len()==3 && args[1]=="check-request" {
        let prepared=PreparedRequest::read(Path::new(&args[2]))?;
        println!("{}",json!({"passed":true,"gpu_requested":false,"workflow":prepared.workflow()?,"nfe":prepared.nfe()?}));
        return Ok(());
    }
    if args.len()!=3 || args[1]!="serve" {return Err("usage: h3-model-run serve SOCKET".into());}
    crate::native::cp_select_device_from_environment()?;
    let socket=PathBuf::from(&args[2]);
    if socket.exists(){return Err("socket already exists".into());}
    if let Some(parent)=socket.parent(){fs::create_dir_all(parent)?;}
    let listener=UnixListener::bind(&socket)?;let _guard=SocketGuard(socket);
    let mut server=Server::default();
    for connection in listener.incoming() {
        let mut stream=connection?;
        let result=(||->Result<Value>{
            let mut line=String::new();BufReader::new(&stream).take(1048577).read_line(&mut line)?;
            if line.len()>1<<20 || !line.ends_with('\n'){return Err("invalid newline-delimited request".into());}
            let value:Value=serde_json::from_str(&line)?;
            server.request(&serde_json::from_value::<Vec<String>>(value["args"].clone())?)
        })();
        let failed=result.is_err();
        let reply=match result {
            Ok(details)=>json!({"ok":true,"error":Value::Null,"details":details}),
            Err(error)=>json!({"ok":false,"error":error.to_string(),"details":{"component":"dit","recoverable":!server.poisoned}}),
        };
        writeln!(stream,"{}",serde_json::to_string(&reply)?)?;stream.flush()?;
        if failed && server.poisoned {return Err("public owner stopped after a CUDA-phase failure".into());}
    }
    Ok(())
}
pub fn main(){if let Err(error)=run(){eprintln!("{error}");std::process::exit(1);}}
