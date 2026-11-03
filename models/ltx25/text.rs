//! Gemma's embedding and 48 decoder layers, composed without later gold activations.
//! This rung admits prepared position/mask tensors; raw text preparation is separate.
use crate::{cuda::{Device,upload_into},gold::{Gold,sha},graph,linear::Linear,
    tensor::{self,Tensor},weights::Checkpoint,Result};
use serde_json::{json,Value};
use std::collections::BTreeMap;

pub(crate) fn weight(checkpoint:&Checkpoint,key:&str,device:&Device)->Result<Tensor>{
    let cache_key=format!("{}\0{key}",checkpoint.path.display());
    if let Some(value)=device.cached_weight(&cache_key){return Ok(value);}
    let (shape,dtype)=checkpoint.weight_layout(key)?;
    let out=Tensor::contiguous_shape(shape,dtype,device)?;
    checkpoint.upload(key,&out,device)?;device.cache_weight(cache_key,&out);Ok(out)
}
fn raw(gold:&Gold,name:&str,device:&Device)->Result<Tensor>{
    let entry=gold.entry(name)?;
    let out=Tensor::contiguous_shape(entry.shape.clone(),entry.dtype.clone(),device)?;
    let data=gold.asset(name)?;upload_into(&out.storage,&data,device)?;Ok(out)
}
fn compare(gold:&Gold,name:&str,value:&Tensor,device:&Device,checks:&mut Vec<Value>)->Result<bool>{
    if !gold.verify{return Ok(true);}
    let actual=sha(&value.download(device)?);let entry=gold.entry(name)?;
    let passed=entry.sha256==actual && entry.shape==value.shape && entry.dtype==value.dtype;
    checks.push(json!({"tensor":name,"passed":passed,"sha256":actual,"expected_sha256":entry.sha256}));Ok(passed)
}

fn token_input(gold:&Gold,checkpoint:&Checkpoint,device:&Device)->Result<(Tensor,Tensor,Vec<i64>)>{
    let prompt=gold.identity["request"]["prompt"].as_str().ok_or("raw prompt")?;
    let input=checkpoint.tokenizer()?.encode(prompt)?;
    let upload=|values:&[i64]|->Result<Tensor>{
        let tensor=Tensor::contiguous_shape(vec![1,values.len()],"torch.int64".into(),device)?;
        let bytes=values.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>();
        upload_into(&tensor.storage,&bytes,device)?;Ok(tensor)
    };
    Ok((upload(&input.ids)?,upload(&input.attention_mask)?,input.attention_mask))
}

/// CPU mask construction and CUDA FP32 RoPE from immutable model constants.
/// The constants are exported model buffers, never request activations.
fn prepare(gold:&Gold,mask:&[i64],device:&Device,checks:&mut Vec<Value>)->Result<Option<Vec<BTreeMap<String,Tensor>>>>{
    if !gold.metadata["gemma.config"].is_object(){return Ok(None);}
    let count=mask.len();
    let positions=Tensor::contiguous_shape(vec![1,count],"torch.int64".into(),device)?;
    tensor::fill(&positions,0.,1.,device.stream)?;
    let pos_float=Tensor::contiguous_shape(vec![1,count],"torch.float32".into(),device)?;
    tensor::copy(&positions,&pos_float,device.stream)?;
    let mut pos_float=pos_float;pos_float.shape.push(1);pos_float.strides.push(0);
    let window=gold.metadata["gemma.config"]["sliding_window"].as_u64().ok_or("Gemma sliding window")? as usize;
    let mut prepared=Vec::new();
    for (prototype,kind) in [(0,"sliding_attention"),(5,"full_attention")]{
        let mut bytes=Vec::with_capacity(count*count);
        for query in 0..count{for key in 0..count{
            bytes.push(u8::from(mask[key]!=0 && key<=query && (kind=="full_attention" || query-key<window)));
        }}
        let causal=Tensor::contiguous_shape(vec![1,1,count,count],"torch.bool".into(),device)?;
        upload_into(&causal.storage,&bytes,device)?;
        let mut inv=raw(gold,&format!("gemma.rope.constants.{kind}_inv_freq"),device)?;
        if inv.shape.len()!=1{return Err("RoPE inverse frequency rank".into());}
        let half=inv.shape[0];let dim=half*2;
        if inv.dtype!="torch.float32"{
            let rounded=Tensor::contiguous_shape(inv.shape.clone(),"torch.float32".into(),device)?;
            tensor::copy(&inv,&rounded,device.stream)?;inv=rounded;
        }
        inv.shape=vec![1,1,half];inv.strides=vec![0,0,1];
        let freq=Tensor::contiguous_shape(vec![1,count,half],"torch.float32".into(),device)?;
        tensor::elementwise(21,&pos_float,Some(&inv),None,&freq,0.,1.,device.stream)?;
        let embedded=Tensor::contiguous_shape(vec![1,count,dim],"torch.float32".into(),device)?;
        for part in 0..2{
            let mut slice=embedded.clone();slice.shape[2]=half;slice.offset=part*half*4;
            tensor::copy(&freq,&slice,device.stream)?;
        }
        let scaling=gold.metadata["gemma.rope.scaling"][kind].as_f64().ok_or("RoPE scaling")? as f32;
        let mut fields=BTreeMap::from([("attention_mask".to_owned(),causal),("position_ids".to_owned(),positions.clone())]);
        for (component,op) in [(0,8),(1,9)]{
            let trig=Tensor::contiguous_shape(embedded.shape.clone(),"torch.float32".into(),device)?;
            tensor::elementwise(op,&embedded,None,None,&trig,0.,1.,device.stream)?;
            let scaled=Tensor::contiguous_shape(trig.shape.clone(),"torch.float32".into(),device)?;
            tensor::elementwise(21,&trig,None,None,&scaled,scaling,1.,device.stream)?;
            let result=Tensor::contiguous_shape(trig.shape.clone(),"torch.bfloat16".into(),device)?;
            tensor::copy(&scaled,&result,device.stream)?;
            fields.insert(format!("position_embeddings.{component}"),result);
        }
        for (key,value) in &fields{
            if !compare(gold,&format!("gemma.layer.{prototype}.kwargs.{key}"),value,device,checks)?{
                return Err(format!("native Gemma preparation differs: {kind} {key}").into());
            }
        }
        prepared.push(fields);
    }
    Ok(Some(prepared))
}

pub fn run(gold:&Gold,checkpoint:&Checkpoint,device:&Device,linear:&Linear)->Result<(Vec<Tensor>,Value)>{
    let _trace=native_core_infra::startup_trace::Span::new("text_encoder",gold.root.to_string_lossy(),0);
    let mut checks=Vec::new();
    let fail=|checks:Vec<Value>|json!({"passed":false,"rung":"gemma","checks":checks});
    let (ids,attention,mask)=token_input(gold,checkpoint,device)?;
    if !compare(gold,"gemma.input_ids",&ids,device,&mut checks)? ||
        !compare(gold,"gemma.attention_mask",&attention,device,&mut checks)?{return Ok((vec![],fail(checks)));}
    let prepared=prepare(gold,&mask,device,&mut checks)?;
    if ids.dtype!="torch.int64" || ids.shape.len()!=2 || ids.shape[0]!=1{return Err("Gemma token input outside port".into());}
    let mut table=weight(checkpoint,"model.embed_tokens.weight",device)?;
    if table.shape.len()!=2{return Err("embedding table must be a matrix".into());}
    let (vocab,dim)=(table.shape[0],table.shape[1]);
    table.shape=vec![1,vocab,dim];table.strides=vec![vocab*dim,dim,1];
    let mut indices=ids.clone();indices.shape.push(dim);indices.strides.push(0);
    let hidden=Tensor::contiguous_shape(vec![1,ids.shape[1],dim],table.dtype.clone(),device)?;
    tensor::index(0,&table,Some(&indices),&hidden,1,device)?;drop(table);
    let scale=Tensor::contiguous_shape(vec![],"torch.float32".into(),device)?;
    let scalar=((dim as f64).sqrt() as f32).to_ne_bytes();upload_into(&scale.storage,&scalar,device)?;
    if !compare(gold,"gemma.embed_scale",&scale,device,&mut checks)?{return Ok((vec![],fail(checks)));}
    let rounded=Tensor::contiguous_shape(vec![],hidden.dtype.clone(),device)?;
    tensor::copy(&scale,&rounded,device.stream)?;
    tensor::elementwise(21,&hidden,Some(&rounded),None,&hidden,0.,1.,device.stream)?;
    if !compare(gold,"gemma.embedding.out",&hidden,device,&mut checks)?{return Ok((vec![],fail(checks)));}
    let mut states=vec![hidden.clone()];let mut hidden=hidden;
    let templates=[0,5].into_iter().map(|i|->Result<Value>{
        Ok(serde_json::from_slice(&std::fs::read(gold.root.join(format!("gemma.layer.{i}.graph.json")))?)?)
    }).collect::<Result<Vec<_>>>()?;
    for layer in 0..48{
        let _layer=native_core_infra::startup_trace::Span::new("text_layer",layer.to_string(),0);
        let kind=usize::from(layer%6==5);let prototype=[0,5][kind];let template=&templates[kind];
        let prefix=format!("gemma.layer.{prototype}");
        if !compare(gold,&format!("gemma.layer.{layer}.args.0"),&hidden,device,&mut checks)?{return Ok((states,fail(checks)));}
        let mut replacements=BTreeMap::from([(format!("{prefix}.args.0"),hidden.clone())]);
        for (name,_) in template["externals"].as_object().ok_or("Gemma externals")?{
            if let Some(suffix)=name.strip_prefix(&format!("{prefix}.weights.")){
                replacements.insert(name.clone(),weight(checkpoint,&format!("model.layers.{layer}.{suffix}"),device)?);
            }else if let Some(suffix)=name.strip_prefix(&format!("{prefix}.buffers.")){
                replacements.insert(name.clone(),weight(checkpoint,&format!("model.layers.{layer}.{suffix}"),device)?);
            }else if let Some(suffix)=name.strip_prefix(&format!("{prefix}.kwargs.")){
                let first=gold.entry(name)?;let actual=gold.entry(&format!("gemma.layer.{layer}.kwargs.{suffix}"))?;
                if first.sha256!=actual.sha256 || first.dtype!=actual.dtype || first.shape!=actual.shape{
                    return Err(format!("Gemma layer {layer} changes prepared {suffix}").into());
                }
                if let Some(prepared)=&prepared{
                    replacements.insert(name.clone(),prepared[kind].get(suffix).ok_or("unported Gemma prepared field")?.clone());
                }
            }
        }
        let inputs=graph::load_externals_replacing(gold,template,device,&replacements)?;
        let (values,report)=graph::execute(template,inputs,device,linear,false)?;
        if report["executed"]!=true{return Err("native Gemma layer did not finish".into());}
        let output=template["output_paths"][""].as_str().ok_or("Gemma layer output")?;
        hidden=values.get(output).ok_or("missing Gemma output")?.clone();
        if !compare(gold,&format!("gemma.layer.{layer}.out"),&hidden,device,&mut checks)?{return Ok((states,fail(checks)));}
        states.push(hidden.clone());
        eprintln!("{}",json!({"event":"gemma-layer-passed","layer":layer}));
    }
    let norm:Value=serde_json::from_slice(&std::fs::read(gold.root.join("gemma.final_norm.graph.json"))?)?;
    let mut replacements=BTreeMap::from([("gemma.final_norm.args.0".into(),hidden)]);
    replacements.insert("gemma.final_norm.weights.weight".into(),weight(checkpoint,"model.norm.weight",device)?);
    let inputs=graph::load_externals_replacing(gold,&norm,device,&replacements)?;
    let (values,_)=graph::execute(&norm,inputs,device,linear,false)?;
    let final_hidden=values.get(norm["output_paths"][""].as_str().ok_or("Gemma norm output")?).ok_or("native norm output")?.clone();
    if !compare(gold,"gemma.final_norm.out",&final_hidden,device,&mut checks)?{return Ok((states,fail(checks)));}
    states[48]=final_hidden;
    for (index,state) in states.iter().enumerate(){
        if !compare(gold,&format!("gemma.hidden_states.{index}"),state,device,&mut checks)?{return Ok((states,fail(checks)));}
    }
    Ok((states,json!({"passed":true,"rung":"gemma","checks":checks,"native_input_preparation":prepared.is_some(),
        "scope":"native tokenizer, embedding and 48 layers; optional native mask/RoPE from model constants"})))
}

pub(crate) fn contexts_ready(gold:&Gold,checkpoint:&Checkpoint,transformer:&Checkpoint,device:&Device,linear:&Linear,
    ready:impl FnOnce()->Result<()>)->Result<(Vec<Tensor>,Value)>{
    let (states,mut report)=run(gold,checkpoint,device,linear)?;
    if report["passed"]!=true{return Ok((vec![],report));}
    ready()?;
    let processor:Value=serde_json::from_slice(&std::fs::read(gold.root.join("processor.graph.json"))?)?;
    let (_,attention,_)=token_input(gold,checkpoint,device)?;
    let mut replacements=BTreeMap::from([("processor.args.1".to_owned(),attention)]);
    for (index,state) in states.into_iter().enumerate(){replacements.insert(format!("processor.args.0.{index}"),state);}
    for (name,_) in processor["externals"].as_object().ok_or("processor externals")?{
        let Some(suffix)=name.strip_prefix("processor.weights.")else{continue;};
        let (ckpt,key)=if let Some(s)=suffix.strip_prefix("feature_extractor."){
            (checkpoint,format!("text_embedding_projection.{s}"))
        }else if let Some(s)=suffix.strip_prefix("video_connector."){
            (transformer,format!("model.diffusion_model.video_embeddings_connector.{s}"))
        }else if let Some(s)=suffix.strip_prefix("audio_connector."){
            (transformer,format!("model.diffusion_model.audio_embeddings_connector.{s}"))
        }else{return Err(format!("unported processor weight {suffix}").into());};
        replacements.insert(name.clone(),weight(ckpt,&key,&device)?);
    }
    let inputs=graph::load_externals_replacing(gold,&processor,&device,&replacements)?;
    let (values,_)=graph::execute(&processor,inputs,&device,&linear,false)?;
    let checks=report["checks"].as_array_mut().ok_or("text checks")?;let mut outputs=Vec::new();
    for component in 0..3{
        let id=processor["output_paths"][format!(".{component}")].as_str().ok_or("processor output")?;
        if !compare(gold,&format!("processor.out.{component}"),values.get(id).ok_or("native context")?,&device,checks)?{
            report["passed"]=json!(false);break;
        }
        outputs.push(values[id].clone());
    }
    report["rung"]=json!("text-contexts");Ok((outputs,report))
}
