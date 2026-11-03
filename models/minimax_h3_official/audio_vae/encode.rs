pub use h3_official_audio_vae::startup_trace;
use h3_official_audio_vae::{Decoder,Tensor,digest};
use serde_json::{json,Value};
use std::{fs,path::PathBuf,ptr,time::Instant};
struct Resident { model:Option<PathBuf>, manifest:Value, encoder:Option<Decoder>, requests:usize }
impl Resident {
 fn new()->Self{Self{model:None,manifest:Value::Null,encoder:None,requests:0}}
 fn execute(&mut self,args:&[String])->Result<Value,String>{
 if args.get(1).is_some_and(|v|v=="preload") {
  if args.len()!=3 || self.encoder.is_some(){return Err("preload requires one unloaded audio reference model".into());}
  let started=Instant::now();let model=fs::canonicalize(&args[2]).map_err(|e|e.to_string())?;
  self.manifest=serde_json::from_slice(&crate::startup_trace::read(model.join("manifest.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
  self.encoder=Some(Decoder::load_encoder(&model)?);self.model=Some(model);
  return Ok(json!({"component":"audio_encoder","model_load_count":1,"requests_completed":0,
      "load_seconds":started.elapsed().as_secs_f64()}));
 }
 if args.len()!=4&&args.len()!=5{return Err("h3-audio-encode MODEL WAVEFORM OUTPUT_DIRECTORY [--validate]".into());}
 let model=fs::canonicalize(&args[1]).map_err(|e|e.to_string())?;let input=PathBuf::from(&args[2]);let out=PathBuf::from(&args[3]);
 let validate=args.get(4).map(|s|s=="--validate").unwrap_or(false);
 if args.len()==5&&!validate{return Err("unsupported encoder argument".into());}
 if !out.is_absolute()||out.exists(){return Err("fresh absolute output directory required".into());}
 fs::create_dir_all(&out).map_err(|e|e.to_string())?;
 let started=Instant::now();
 if self.model.as_ref().is_some_and(|p|p!=&model){return Err("resident audio encoder model changed".into());}
 let first=self.encoder.is_none();
 if first {
  self.manifest=serde_json::from_slice(&crate::startup_trace::read(model.join("manifest.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
  self.encoder=Some(Decoder::load_encoder(&model)?);self.model=Some(model);
 }
 let manifest=&self.manifest;
 let encoder=self.encoder.as_mut().ok_or("missing resident encoder")?;
 let load_seconds=if first{started.elapsed().as_secs_f64()}else{0.};
 let bytes=crate::startup_trace::read(input).map_err(|e|e.to_string())?;
 if bytes.chunks_exact(4).any(|b|!f32::from_le_bytes(b.try_into().unwrap()).is_finite()){return Err("nonfinite audio input".into());}
 if bytes.is_empty()||bytes.len()%8!=0{return Err("expected contiguous FP32 stereo waveform".into());}
 let samples=(bytes.len()/8) as i64;let mut padded=Vec::new();
 let aligned=if validate{samples}else{((samples+799)/800)*800};
 let generic=self.manifest["input"]["shape"][2].as_i64()!=Some(samples);
 if generic && validate {return Err("per-node validation requires original shape".into());}
 // Always select, including B -> A when A is the architecture template.
 if !validate{encoder.select_encoder_samples(aligned)?;}
 if aligned!=samples {
  if validate{return Err("per-node validation requires the original captured shape; generic shapes use independent output comparison".into());}
  
  padded.resize((aligned*8) as usize,0u8);
  for channel in 0..2{let from=channel*samples as usize*4;let to=channel*aligned as usize*4;padded[to..to+samples as usize*4].copy_from_slice(&bytes[from..from+samples as usize*4]);}
 }
 let tensor=Tensor::from_bytes(encoder.input_shape(),if aligned!=samples{&padded}else{&bytes})?;
 let computed=unsafe{encoder.run(tensor.pointer(),ptr::null_mut(),if validate{Some(&out)}else{None})}?;
 let posterior=computed.download(ptr::null_mut())?;
 let shape=&computed.shape;
 if shape.len()!=3||shape[..2]!=[2,32]{return Err("encoder posterior shape".into());}
 let read_config=|name:&str|->Result<Vec<f32>,String>{let a=manifest[name].as_array().ok_or("missing latent normalization")?;if a.len()!=32{return Err("latent normalization width".into());}a.iter().map(|v|v.as_f64().map(|x|x as f32).filter(|v|v.is_finite()).ok_or("nonfinite config".into())).collect()};
 let mean=read_config("latents_mean")?;let std=read_config("latents_std")?;
 if std.iter().any(|v|*v<=0.){return Err("invalid latent std".into());}
 let frames=shape[2] as usize;let mut condition=Vec::with_capacity(posterior.len());let mut nonfinite=0;
 for b in 0..2 {for t in 0..frames{for c in 0..32{
  let i=((b*32+c)*frames+t)*4;let x=f32::from_le_bytes(posterior[i..i+4].try_into().unwrap());
  let normalized=(x-mean[c])/std[c];if !normalized.is_finite(){nonfinite+=1;}condition.extend_from_slice(&normalized.to_le_bytes());
 }}}
 if nonfinite!=0{return Err("nonfinite posterior".into());}
 let maps=fs::read_to_string("/proc/self/maps").map_err(|e|e.to_string())?;
 if maps.lines().any(|s|["libtorch","libaten","libpython","libc10"].iter().any(|v|s.to_lowercase().contains(v))){return Err("forbidden native dependency".into());}
 fs::write(out.join("posterior.raw"),&posterior).map_err(|e|e.to_string())?;
 fs::write(out.join("condition.raw"),&condition).map_err(|e|e.to_string())?;
 self.requests+=1;
 let report=json!({"resident":{"model_load_count":1,"request_load_seconds":load_seconds,"requests_completed":self.requests},"passed":true,"scope":"native audio reference encoder from normalized FP32 stereo","gold_activations_read":validate,
  "manifest_sha256":encoder.graph_manifest_sha(),"generic_shape_inferred":generic,"execution_input_shape":encoder.input_shape(),"input_sha256":digest(&bytes),"posterior_sha256":digest(&posterior),
  "condition_sha256":digest(&condition),"condition_shape":[2*frames,32],"nonfinite":0,"nodes":encoder.graph_node_count(),"wall_seconds":started.elapsed().as_secs_f64()});
 fs::write(out.join("report.json"),serde_json::to_vec_pretty(&report).unwrap()).map_err(|e|e.to_string())?;Ok(report)
 }
}
fn run()->Result<(),String>{
 let args:Vec<_>=std::env::args().collect();
 let mut resident=Resident::new();
 if args.len()==3 && args[1]=="serve" {
  use std::io::{BufRead,BufReader,Read,Write};
  let listener=std::os::unix::net::UnixListener::bind(&args[2]).map_err(|e|e.to_string())?;
  for connection in listener.incoming(){
   let mut stream=connection.map_err(|e|e.to_string())?;
   stream.set_read_timeout(Some(std::time::Duration::from_secs(30))).map_err(|e|e.to_string())?;
   let result=(||->Result<Value,String>{
    let mut line=String::new();BufReader::new((&stream).take(65537)).read_line(&mut line).map_err(|e|e.to_string())?;
    if line.len()>65536 || !line.ends_with('\n'){return Err("bounded newline-delimited request required".into());}
    let input:Value=serde_json::from_str(&line).map_err(|e|e.to_string())?;
    let mut argv=vec!["h3-audio-encode".into()];
    argv.extend(serde_json::from_value::<Vec<String>>(input["args"].clone()).map_err(|e|e.to_string())?);
    if argv.len()!=4 && !(argv.len()==3 && argv[1]=="preload"){return Err("resident audio encoder requires generation or preload".into());}
    resident.execute(&argv)
   })();
   let response=match &result{Ok(report)=>json!({"ok":true,"details":report}),Err(error)=>json!({"ok":false,"error":error})};
   writeln!(stream,"{response}").map_err(|e|e.to_string())?;stream.flush().map_err(|e|e.to_string())?;
   // The supervisor reaps every peer on an execution failure.
   result?;
  }
  Ok(())
 } else {println!("{}",resident.execute(&args)?);Ok(())}
}
fn main(){if let Err(error)=run(){eprintln!("{error}");std::process::exit(1);}}
