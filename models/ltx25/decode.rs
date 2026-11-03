//! Native audio and diffusion-video decoding from generated final latents.
//! Parameters are immutable converted checkpoint assets; no activation fallback.
use crate::{cuda::Device,gold::{Gold,sha},graph,linear::Linear,tensor::{self,Tensor},Result};
use serde_json::{json,Value};
use std::{collections::BTreeMap,path::Path};

fn check(aux:&Gold,name:&str,value:&Tensor,device:&Device,checks:&mut Vec<Value>)->Result<()>{
    if !aux.verify{return Ok(());}
    let entry=aux.entry(name)?;let actual=sha(&value.download(device)?);let passed=entry.sha256==actual;
    checks.push(json!({"tensor":name,"passed":passed,"sha256":actual,"expected_sha256":entry.sha256}));
    if !passed{return Err(format!("native decoder differs: {name}").into());}Ok(())
}
fn run(aux:&Gold,name:&str,input:&Tensor,seed:Option<u64>,device:&Device,linear:&Linear)->Result<Vec<Tensor>>{
    let mut program:Value=serde_json::from_slice(&std::fs::read(aux.root.join(format!("{name}.graph.json")))?)?;
    if let Some(seed)=seed{
        for node in program["nodes"].as_array_mut().ok_or("decoder nodes")?{
            if node["op"]=="ltx25.randn"{node["kwargs"]["seed"]=json!(seed);}
        }
    }
    let replacements=BTreeMap::from([(format!("{name}.args.0"),input.clone())]);
    for field in graph::external_names(&program)?{
        if (field.contains(".args.") || field.contains(".kwargs.")) && !replacements.contains_key(&field){
            return Err(format!("decoder activation was not bound natively: {field}").into());
        }
    }
    let tensors=graph::load_externals_replacing(aux,&program,device,&replacements)?;
    let (values,report)=graph::execute(&program,tensors,device,linear,false)?;
    if report["executed"]!=true{return Err("native decoder did not complete".into());}
    program["outputs"].as_array().ok_or("decoder outputs")?.iter().map(|id|
        values.get(id.as_str().ok_or("decoder output id")?).cloned().ok_or_else(||"decoder result missing".into())).collect()
}
pub fn audio(aux:&Gold,latent:&Tensor,device:&Device,linear:&Linear)->Result<(Tensor,Value)>{
    let mut checks=Vec::new();let decoded=run(aux,"audio.decoder",latent,None,device,linear)?.remove(0);
    check(aux,"audio.decoder.out",&decoded,device,&mut checks)?;
    let waveform=run(aux,"audio.vocoder",&decoded,None,device,linear)?.remove(0);
    check(aux,"audio.vocoder.out",&waveform,device,&mut checks)?;
    if waveform.shape.len()!=3 || waveform.shape[0]!=1 || waveform.shape[1]!=2{return Err("native stereo waveform extent".into());}
    let out=Tensor::contiguous_shape(vec![2,waveform.shape[2]],"torch.float32".into(),device)?;
    tensor::copy(&waveform,&out,device.stream)?;check(aux,"audio.waveform",&out,device,&mut checks)?;
    Ok((out,json!({"passed":true,"sampling_rate":aux.metadata["sampling_rate"],"checks":checks})))
}
pub fn video(aux:&Gold,latent:&Tensor,seed:u64,device:&Device,linear:&Linear)->Result<(Vec<Tensor>,Value)>{
    let mut checks=Vec::new();let outputs=run(aux,"video.pipeline",latent,Some(seed),device,linear)?;
    for (index,value) in outputs.iter().enumerate(){check(aux,&format!("video.pixels.{index}"),value,device,&mut checks)?;}
    Ok((outputs,json!({"passed":true,"checks":checks})))
}
pub fn write_raw(output:&Path,name:&str,value:&Tensor,device:&Device)->Result<Value>{
    let raw=value.download(device)?;let path=output.join(name);std::fs::write(&path,&raw)?;
    Ok(json!({"path":path,"dtype":value.dtype,"shape":value.shape,"sha256":sha(&raw),"bytes":raw.len()}))
}
