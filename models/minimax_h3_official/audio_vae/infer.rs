//! Serving execution mode: reads weights and caller input, never gold data.
pub use h3_official_audio_vae::startup_trace;
#[path="../support/debug_audit.rs"]
pub mod debug_audit;
mod audio_pool_policy;
use h3_official_audio_vae::{Decoder,Tensor,digest};
use serde_json::json;
use std::{fs,path::PathBuf,ptr,time::Instant};

fn ordinary_main(){
    let args:Vec<String>=std::env::args().collect();
    if args.len()!=5 {
        eprintln!("usage: h3-audio-infer MODEL_ROOT NORMALIZED_FP32_INPUT OUTPUT_FP32 REPORT_JSON");
        std::process::exit(2);
    }
    let model=PathBuf::from(&args[1]);let input=PathBuf::from(&args[2]);let output=PathBuf::from(&args[3]);let report_path=PathBuf::from(&args[4]);
    let started=Instant::now();
    let mut report=json!({"schema":"h3-audio-native-infer-v1","passed":false,"component":"audio-vae",
        "model_root":model,"input_file":input,"output_file":output,
        "input_dtype":"float32 normalized channel-major",
        "output_dtype":"float32 channel-major","sampling_rate":32000,"gold_activations_read":false});
    let result=(||->Result<(),String>{
            let pool_enabled=audio_pool_policy::read()?;
            report["allocation_pool_enabled"]=json!(pool_enabled);
        crate::debug_audit::begin()?;
        let model_bytes=crate::startup_trace::read(model.join("manifest.json")).map_err(|e|e.to_string())?;
        report["model_manifest_sha256"]=json!(digest(&model_bytes));
        let input_bytes=fs::metadata(&input).map_err(|e|e.to_string())?.len();
        if input_bytes==0||input_bytes%256!=0{return Err("audio input must be contiguous FP32 [2,32,T] (positive multiple of 256 bytes)".into());}
        let mut decoder=Decoder::load(&model)?;
        decoder.select_tokens(&model,(input_bytes/256) as i64,crate::debug_audit::enabled())?;
        decoder.set_allocation_pool_enabled(pool_enabled);
        let input_shape=decoder.input_shape().to_vec();let output_shape=decoder.output_shape().to_vec();
        report["input_shape"]=json!(input_shape);report["output_shape"]=json!(output_shape);
        let bytes=crate::startup_trace::read(&input).map_err(|e|e.to_string())?;
        if bytes.len()!=input_shape.iter().product::<i64>() as usize*4{return Err("audio input bytes differ from selected model shape".into());}
        if bytes.chunks_exact(4).any(|v|!f32::from_le_bytes(v.try_into().unwrap()).is_finite()){return Err("nonfinite audio input".into());}
        report["input_sha256"]=json!(digest(&bytes));
        let input_tensor=Tensor::from_bytes(&input_shape,&bytes)?;
        report["load_seconds"]=json!(started.elapsed().as_secs_f64());
        let forward=Instant::now();
        let tensor=unsafe{decoder.run(input_tensor.pointer(),ptr::null_mut(),None)}?;
        let audio=tensor.download(ptr::null_mut())?;
        report["forward_seconds"]=json!(forward.elapsed().as_secs_f64());
        if audio.len()!=output_shape.iter().product::<i64>() as usize*4{return Err("audio output shape mismatch".into());}
        let nonfinite=audio.chunks_exact(4).filter(|v|!f32::from_le_bytes((*v).try_into().unwrap()).is_finite()).count();
        report["nonfinite"]=json!(nonfinite);
        if nonfinite!=0{return Err("nonfinite audio output".into());}
        report["output_sha256"]=json!(digest(&audio));
        report["output_bytes"]=json!(audio.len());
        report["computed_nodes"]=json!(decoder.graph_node_count());
            report["dynamic_graph"]=decoder.dynamic_graph_report();
            report["allocation_pool"]=decoder.allocation_pool_report();
            report["allocation_pool_plans"]=if crate::debug_audit::enabled(){json!(decoder.plans)}else{serde_json::Value::Null};
        let maps=fs::read_to_string("/proc/self/maps").map_err(|e|e.to_string())?;
        let forbidden:Vec<_>=maps.lines().filter(|s|["libtorch","libpython","libaten","libc10"].iter().any(|v|s.to_lowercase().contains(v))).collect();
        report["forbidden_mapped_libraries"]=json!(forbidden);
        if !forbidden.is_empty(){return Err("forbidden native dependency".into());}
        if let Some(parent)=output.parent(){fs::create_dir_all(parent).map_err(|e|e.to_string())?;}
        fs::write(&output,&audio).map_err(|e|e.to_string())?;
        report["passed"]=json!(true);
        Ok(())
    })();
    if let Err(error)=&result{report["error"]=json!(error);}
    report["wall_seconds"]=json!(started.elapsed().as_secs_f64());
    let encoded=serde_json::to_vec_pretty(&report).unwrap();
    if let Some(parent)=report_path.parent(){fs::create_dir_all(parent).unwrap();}
    fs::write(&report_path,&encoded).unwrap();
    println!("{}",report);
    if result.is_err(){std::process::exit(1);}
}

// Added only to an isolated copy of the accepted audio infer entrypoint.
// Decoder::load/run, Tensor and the accepted native libraries are unchanged.
mod audio_health;
mod resident_server {
    use super::*;
    use serde_json::Value;
    use std::io::{BufRead,BufReader,Write};
    use std::os::unix::net::UnixListener;
    use std::path::Path;

    extern "C" {
        fn cudaMemGetInfo(free:*mut usize,total:*mut usize)->i32;
        fn cudaStreamSynchronize(stream:*mut std::ffi::c_void)->i32;
    }
    fn memory()->Result<Value,String>{
        let(mut free,mut total)=(0usize,0usize);
        let code=unsafe{cudaMemGetInfo(&mut free,&mut total)};
        if code!=0{return Err(format!("cudaMemGetInfo: {code}"));}
        Ok(json!({"free_bytes":free,"total_bytes":total,"device_used_bytes":total-free}))
    }
    fn synchronize()->Result<(),String>{
        let code=unsafe{cudaStreamSynchronize(ptr::null_mut())};
        if code==0{Ok(())}else{Err(format!("cudaStreamSynchronize: {code}"))}
    }
    struct State {
        decoder:Option<Decoder>,model:Option<PathBuf>,manifest_sha:Option<String>,
        model_load_count:usize,requests_completed:usize,
        health:Option<audio_health::Health>,
        pool_override:Option<bool>,
    }
    impl State {fn new()->Self{Self{decoder:None,model:None,manifest_sha:None,model_load_count:0,requests_completed:0,health:None,pool_override:None}}}
    fn failure(message:String,state:&State)->Value{
        json!({"ok":false,"error":message,"details":{"component":"audio_decode",
            "requests_completed":state.requests_completed,"model_load_count":state.model_load_count}})
    }
    fn pool_mode_enabled(report:&Value)->bool {report["allocation_pool_enabled"]==true}
    fn request(args:&[String],state:&mut State)->Value{
        if args.first().is_some_and(|v|v=="allocation-pool") {
            let result=(||->Result<Value,String>{
                if args.len()!=2 || !matches!(args[1].as_str(),"0"|"default") {
                    return Err("allocation-pool requires 0 or default".into());
                }
                let enabled=if args[1]=="0"{false}else{audio_pool_policy::read()?};
                let mut released=0;
                if let Some(decoder)=state.decoder.as_mut(){
                    if !enabled{released=decoder.release_cached_intermediates()?;}
                    decoder.set_allocation_pool_enabled(enabled);
                }
                state.pool_override=if args[1]=="0"{Some(false)}else{None};
                let cached=state.decoder.as_ref().map_or(0,|d|d.allocation_pool_report()["cached_bytes"].as_u64().unwrap_or(0));
                Ok(json!({"component":"audio_pool_policy","enabled":enabled,
                    "released_cached_bytes":released,"cached_bytes":cached,
                    "model_load_count":state.model_load_count,"requests_completed":state.requests_completed,
                    "stream_synchronized_before_release":true}))
            })();
            return match result{Ok(details)=>json!({"ok":true,"details":details}),Err(e)=>failure(e,state)};
        }
        if args.first().is_some_and(|v|v=="preload") {
            let result=(||->Result<Value,String>{
                if args.len()!=2 || state.decoder.is_some(){return Err("preload requires one unloaded audio model".into());}
                let started=Instant::now();let model=fs::canonicalize(&args[1]).map_err(|e|e.to_string())?;
                let decoder=Decoder::load(&model)?;synchronize()?;
                state.manifest_sha=Some(decoder.graph_manifest_sha().to_string());state.model=Some(model);
                state.decoder=Some(decoder);state.model_load_count=1;
                Ok(json!({"component":"audio_vae","model_load_count":1,"requests_completed":0,
                    "load_seconds":started.elapsed().as_secs_f64()}))
            })();
            return match result {Ok(details)=>json!({"ok":true,"details":details}),Err(e)=>failure(e,state)};
        }
        if args.len()!=4{return failure("expected MODEL_ROOT NORMALIZED_FP32_INPUT OUTPUT_FP32 REPORT_JSON".into(),state);}
        let model=PathBuf::from(&args[0]);let input=PathBuf::from(&args[1]);
        let output=PathBuf::from(&args[2]);let report_path=PathBuf::from(&args[3]);
        if let Err(error)=crate::debug_audit::begin(){return failure(error,state);}
        let started=Instant::now();let mut request_load_seconds=0f64;
        let mut report=json!({"schema":"h3-audio-native-infer-v1","passed":false,"component":"audio-vae",
            "model_root":model,"input_file":input,"output_file":output,
            "input_dtype":"float32 normalized channel-major",
            "output_dtype":"float32 channel-major","sampling_rate":32000,"gold_activations_read":false});
        let result=(||->Result<(),String>{
            let pool_enabled=match state.pool_override{Some(value)=>value,None=>audio_pool_policy::read()?};
            report["allocation_pool_enabled"]=json!(pool_enabled);
            let input_bytes=fs::metadata(&input).map_err(|e|e.to_string())?.len();
            if input_bytes==0||input_bytes%256!=0{return Err("audio input must be contiguous FP32 [2,32,T] (positive multiple of 256 bytes)".into());}
            let canonical=model.canonicalize().map_err(|e|e.to_string())?;
            if state.decoder.is_none(){
                let loading=Instant::now();state.decoder=Some(Decoder::load(&model)?);
                synchronize()?;request_load_seconds=loading.elapsed().as_secs_f64();state.model_load_count+=1;
            }
            state.decoder.as_mut().unwrap().select_tokens(&canonical,(input_bytes/256) as i64,crate::debug_audit::enabled())?;
            let decoder=state.decoder.as_mut().ok_or("missing resident Decoder")?;
            let model_sha=decoder.graph_manifest_sha().to_string();
            state.model=Some(canonical);state.manifest_sha=Some(model_sha.clone());
            report["model_manifest_sha256"]=json!(model_sha);
            report["dynamic_graph"]=decoder.dynamic_graph_report();
            decoder.set_allocation_pool_enabled(pool_enabled);
            let input_shape=decoder.input_shape().to_vec();let output_shape=decoder.output_shape().to_vec();
            report["input_shape"]=json!(input_shape);report["output_shape"]=json!(output_shape);
            let bytes=crate::startup_trace::read(&input).map_err(|e|e.to_string())?;
            if bytes.len()!=input_shape.iter().product::<i64>() as usize*4{return Err("audio input bytes differ from selected model shape".into());}
            if bytes.chunks_exact(4).any(|v|!f32::from_le_bytes(v.try_into().unwrap()).is_finite()){return Err("nonfinite audio input".into());}
            report["input_sha256"]=crate::debug_audit::sha(||digest(&bytes));
            let input_tensor=Tensor::from_bytes(&input_shape,&bytes)?;
            // Preserve the original load_seconds boundary (model lookup plus
            // input read/upload); request_load_seconds separately measures only
            // the one-time Decoder load and is exactly zero on warm requests.
            report["load_seconds"]=json!(started.elapsed().as_secs_f64());
            let forward=Instant::now();
            let tensor=unsafe{decoder.run(input_tensor.pointer(),ptr::null_mut(),None)}?;
            report["output_shape"]=json!(output_shape);
            let health_result=(||->Result<(),String>{
                if state.health.is_none(){state.health=Some(audio_health::Health::new()?);}
                state.health.as_mut().unwrap().enqueue(tensor.pointer(),output_shape.iter().product::<i64>()as usize)
            })();
            let audio=tensor.download(ptr::null_mut())?;
            report["output_health"]=match health_result{
                Ok(())=>state.health.as_ref().unwrap().collect(),
                Err(error)=>json!({"error":error}),
            };
            report["forward_seconds"]=json!(forward.elapsed().as_secs_f64());
            if audio.len()!=output_shape.iter().product::<i64>() as usize*4{return Err("audio output shape mismatch".into());}
            let nonfinite=audio.chunks_exact(4).filter(|v|!f32::from_le_bytes((*v).try_into().unwrap()).is_finite()).count();
            report["nonfinite"]=json!(nonfinite);
            if nonfinite!=0{return Err("nonfinite audio output".into());}
            report["output_sha256"]=crate::debug_audit::sha(||digest(&audio));report["output_bytes"]=json!(audio.len());
            report["computed_nodes"]=json!(decoder.graph_node_count());
            report["dynamic_graph"]=decoder.dynamic_graph_report();
            report["allocation_pool"]=decoder.allocation_pool_report();
            report["allocation_pool_plans"]=if crate::debug_audit::enabled(){json!(decoder.plans)}else{serde_json::Value::Null};
            let maps=if crate::debug_audit::enabled(){fs::read_to_string("/proc/self/maps").map_err(|e|e.to_string())?}else{String::new()};
            let forbidden:Vec<_>=maps.lines().filter(|s|["libtorch","libpython","libaten","libc10"].iter().any(|v|s.to_lowercase().contains(v))).collect();
            report["forbidden_mapped_libraries"]=if crate::debug_audit::enabled(){json!(forbidden)}else{Value::Null};
            if !forbidden.is_empty(){return Err("forbidden native dependency".into());}
            if let Some(parent)=output.parent(){fs::create_dir_all(parent).map_err(|e|e.to_string())?;}
            fs::write(&output,&audio).map_err(|e|e.to_string())?;
            // The original run releases consumed intermediates. The download
            // synchronizes the stream, then these last two owned buffers drop.
            synchronize()?;
            drop(tensor);drop(input_tensor);
            report["passed"]=json!(true);
            Ok(())
        })();
        let mut error=result.err();
        if let Some(decoder)=state.decoder.as_ref(){report["allocation_pool"]=decoder.allocation_pool_report();}
        // Error paths also drain queued work before the next request or shutdown.
        if let Err(e)=synchronize(){if error.is_none(){error=Some(e);}}
        let used=match memory(){Ok(v)=>v,Err(e)=>{if error.is_none(){error=Some(e);}Value::Null}};
        if error.is_none(){state.requests_completed+=1;}
        report["passed"]=json!(error.is_none());
        if let Some(e)=&error{report["error"]=json!(e);}
        report["debug_audit"]=crate::debug_audit::record();
        report["wall_seconds"]=json!(started.elapsed().as_secs_f64());
        report["resident"]=json!({"model_load_count":state.model_load_count,
            "requests_completed":state.requests_completed,"request_load_seconds":request_load_seconds,
            "memory_after_scratch_release":used,"stream_synchronized_before_release":true,
            "model_reloaded":false,"allocation_pool":report["allocation_pool"],"dynamic_graph":report["dynamic_graph"]});
        let save=(||->Result<(),String>{
            if let Some(parent)=report_path.parent(){fs::create_dir_all(parent).map_err(|e|e.to_string())?;}
            fs::write(&report_path,serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?).map_err(|e|e.to_string())
        })();
        if let Err(e)=save{if error.is_none(){error=Some(e);}}
        json!({"ok":error.is_none(),"error":error,"details":{
            "component":"audio_decode","requests_completed":state.requests_completed,
            "model_load_count":state.model_load_count,"request_load_seconds":request_load_seconds,"debug_audit":crate::debug_audit::record(),
            "request_wall_seconds":started.elapsed().as_secs_f64(),"memory":used,
            "stream_synchronized_before_release":true,
            "scratch_released":report["allocation_pool"]["live_buffers"]==0,
            "scratch_release_kind":if pool_mode_enabled(&report){"returned_to_exact_size_pool"}else{"cudaFree"},
            "allocation_pool_enabled":report["allocation_pool_enabled"],"allocation_pool":report["allocation_pool"],
            "model_manifest_sha256":state.manifest_sha,"report_file":report_path,
            "output_sha256":report["output_sha256"],"output_bytes":report["output_bytes"],
            "gold_activations_read":false,"dynamic_graph":report["dynamic_graph"]}})
    }
    pub fn serve(path:&Path)->Result<(),String>{
        if path.exists(){return Err(format!("socket path already exists: {}",path.display()));}
        if let Some(parent)=path.parent(){fs::create_dir_all(parent).map_err(|e|e.to_string())?;}
        let listener=UnixListener::bind(path).map_err(|e|e.to_string())?;
        let mut state=State::new();
        eprintln!("audio resident ready {}",path.display());
        for connection in listener.incoming(){
            let mut socket=connection.map_err(|e|e.to_string())?;
            let mut line=String::new();
            let count=BufReader::new(socket.try_clone().map_err(|e|e.to_string())?).read_line(&mut line).map_err(|e|e.to_string())?;
            if count==0{continue;}
            let parsed=serde_json::from_str::<Value>(&line).map_err(|e|e.to_string()).and_then(|v|{
                v["args"].as_array().ok_or("missing args array".to_string())?.iter().map(|x|
                    x.as_str().map(str::to_owned).ok_or("args must be strings".to_string())).collect::<Result<Vec<_>,_>>()
            });
            let response=match parsed{Ok(args)=>request(&args,&mut state),Err(e)=>failure(e,&state)};
            writeln!(socket,"{}",response).map_err(|e|e.to_string())?;
            socket.flush().map_err(|e|e.to_string())?;
            // A failed numerical/IO request stops this bounded experiment; never
            // silently reload or retry a model behind a successful response.
            if response["ok"]!=true{return Err(response["error"].as_str().unwrap_or("audio request failed").to_string());}
        }
        Ok(())
    }
}

fn main(){
    let args:Vec<String>=std::env::args().collect();
    if args.len()==5&&args[1]=="shape"{
        let result=(||->Result<(),String>{let t=args[3].parse::<i64>().map_err(|e|e.to_string())?;
            let graph=h3_official_audio_vae::describe_generic_graph(std::path::Path::new(&args[2]),t)?;
            fs::write(&args[4],serde_json::to_vec_pretty(&graph).map_err(|e|e.to_string())?).map_err(|e|e.to_string())})();
        if let Err(e)=result{eprintln!("{e}");std::process::exit(1);}
    }else if args.len()==3&&args[1]=="serve"{
        if let Err(e)=resident_server::serve(std::path::Path::new(&args[2])){eprintln!("{e}");std::process::exit(1);}
    }else{ordinary_main();}
}
