//! Forty-eight AV blocks composed from the validated eager layer graph.
//! This rung starts from prepared block inputs. It is deliberately called
//! `backbone`, not `forward`: input embedding and the output head are separate.
use crate::{cuda::{Device,upload_into},gold::{Gold,sha},graph,linear::Linear,
    tensor::{Tensor,contiguous_strides},weights::Checkpoint,Result};
use serde_json::{json,Value};
use std::collections::BTreeMap;

pub fn execute(gold:&Gold,template:&Value,checkpoint:&Checkpoint,stage:usize,forward:usize,
    inputs:BTreeMap<String,Tensor>,device:&Device,linear:&Linear)->Result<(BTreeMap<String,Tensor>,Value)>{
    execute_range(gold,template,checkpoint,stage,forward,inputs,device,linear,0..48)
}

pub fn execute_range(gold:&Gold,template:&Value,checkpoint:&Checkpoint,stage:usize,forward:usize,
    mut inputs:BTreeMap<String,Tensor>,device:&Device,linear:&Linear,blocks:std::ops::Range<usize>)
    ->Result<(BTreeMap<String,Tensor>,Value)>{
    if blocks.start>=blocks.end || blocks.end>48{return Err("invalid LTX block partition".into());}
    let count=blocks.len();
    let prefix=format!("block.{stage}.0.0");
    let video_key=format!("{prefix}.kwargs.video.x");let audio_key=format!("{prefix}.kwargs.audio.x");
    let find_output=|modality:usize|->Result<String>{
        let expected=&gold.entry(&format!("{prefix}.out.{modality}.x"))?.sha256;
        for name in template["outputs"].as_array().ok_or("template outputs")?{
            let name=name.as_str().ok_or("output tensor name")?;
            if gold.entry(name)?.sha256==*expected{return Ok(name.into());}
        }
        Err("template has no modality output".into())
    };
    let video_out=find_output(0)?;let audio_out=find_output(1)?;
    let mut checks=Vec::new();
    for block in blocks{
        let setup=graph::host_profiling().then(std::time::Instant::now);
        let mut tensors=inputs.clone();
        for (name,spec) in template["externals"].as_object().ok_or("graph externals")?{
            let Some(suffix)=name.strip_prefix(&format!("{prefix}.weights."))else{continue;};
            let key=format!("model.diffusion_model.transformer_blocks.{block}.{suffix}");
            let info=&checkpoint.header[&key];
            let dtype=match info["dtype"].as_str(){Some("BF16")=>"torch.bfloat16",Some("F32")=>"torch.float32",
                _=>return Err(format!("unported weight dtype for {key}").into())};
            if spec["dtype"]!=dtype{return Err(format!("weight dtype differs: {key}").into());}
            let shape=info["shape"].as_array().ok_or("checkpoint weight shape")?.iter()
                .map(|d|d.as_u64().map(|n|n as usize).ok_or("weight dimension")).collect::<std::result::Result<Vec<_>,_>>()?;
            if json!(shape)!=spec["shape"]{return Err(format!("weight shape differs: {key}").into());}
            let expected=if block==0{Some(gold.entry(name)?.sha256.as_str())}else{None};
            let cache_key=format!("{}\0{key}",checkpoint.path.display());
            let value=if let Some(value)=device.cached_weight(&cache_key){value}else{
                let data=checkpoint.tensor(&key,expected)?;
                let value=Tensor::allocate(shape.clone(),contiguous_strides(&shape),dtype.into(),&device)?;
                upload_into(&value.storage,&data,&device)?;device.cache_weight(cache_key,&value);value
            };
            if value.shape!=shape || value.dtype!=dtype{return Err("cached model parameter layout differs".into());}
            tensors.insert(name.clone(),value);
        }
        // Validate branch-independent context/position/timestep identity before
        // reusing a block template. No later block activation is loaded.
        for name in inputs.keys().filter(|n|*n!=&video_key && *n!=&audio_key){
            let suffix=name.strip_prefix(&prefix).ok_or("template prefix")?;
            let first=gold.entry(&format!("block.{stage}.{forward}.0{suffix}"))?;
            let later=gold.entry(&format!("block.{stage}.{forward}.{block}{suffix}"))?;
            if first.sha256!=later.sha256 || first.shape!=later.shape || first.dtype!=later.dtype{
                return Err(format!("block {block} changes prepared input {suffix}").into());
            }
        }
        if let Some(t)=setup{graph::host_record("backbone.setup",t);}
        let (values,_) = graph::execute(&template,tensors,&device,&linear,false)?;
        for (modality,key,out) in [(0,&video_key,&video_out),(1,&audio_key,&audio_out)]{
            let value=values.get(out).ok_or("missing native block output")?.clone();
            let actual=if gold.verify{sha(&value.download(&device)?)}else{String::new()};
            let expected=&gold.entry(&format!("block.{stage}.{forward}.{block}.out.{modality}.x"))?.sha256;
            let passed=!gold.verify || actual==*expected;
            checks.push(json!({"block":block,"modality":modality,"passed":passed,
                "sha256":actual,"expected_sha256":expected}));
            if !passed{return Ok((inputs,json!({"passed":false,"rung":"backbone","checks":checks})));}
            inputs.insert(key.clone(),value);
        }
        let synced=graph::host_profiling().then(std::time::Instant::now);
        device.sync()?;
        if let Some(t)=synced{graph::host_record("backbone.sync",t);}
    }
    Ok((inputs,json!({"passed":checks.len()==count*2,"rung":"backbone","checks":checks,
        "device_uuid":device.uuid()?,"device_index":device.index})))
}
