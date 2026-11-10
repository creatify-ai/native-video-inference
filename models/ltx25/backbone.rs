//! Forty-eight AV blocks composed from the validated eager layer graph.
//! This rung starts from prepared block inputs. It is deliberately called
//! `backbone`, not `forward`: input embedding and the output head are separate.
use crate::{cuda::{Device,upload_into},gold::{Gold,sha},graph,linear::Linear,
    tensor::{Tensor,contiguous_strides},weights::Checkpoint,Result};
use serde_json::{json,Value};
use std::collections::BTreeMap;

/// One block weight as an external of the block program, checked against the checkpoint header.
struct Binding {name:String,key:String,cache_key:String,shape:Vec<usize>,dtype:&'static str}
thread_local!{
    static BINDINGS:std::cell::RefCell<BTreeMap<(std::path::PathBuf,String,usize),std::rc::Rc<Vec<Binding>>>>=
        const{std::cell::RefCell::new(BTreeMap::new())};
    static VALIDATED:std::cell::RefCell<std::collections::BTreeSet<(std::path::PathBuf,usize,usize,usize)>>=
        const{std::cell::RefCell::new(std::collections::BTreeSet::new())};
}
/// `block`'s weights for the block program, resolved once per checkpoint, stage and block (the
/// names, header checks and cache keys; the tensors themselves come from the device cache).
fn bindings(template:&Value,checkpoint:&Checkpoint,prefix:&str,block:usize)->Result<std::rc::Rc<Vec<Binding>>>{
    let id=(checkpoint.path.clone(),prefix.to_owned(),block);
    if let Some(value)=BINDINGS.with(|b|b.borrow().get(&id).cloned()){return Ok(value);}
    let mut values=Vec::new();
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
        values.push(Binding{name:name.clone(),cache_key:format!("{}\0{key}",checkpoint.path.display()),key,shape,dtype});
    }
    let value=std::rc::Rc::new(values);
    BINDINGS.with(|b|b.borrow_mut().insert(id,value.clone()));Ok(value)
}

pub fn execute(gold:&Gold,template:&Value,checkpoint:&Checkpoint,stage:usize,forward:usize,
    inputs:BTreeMap<String,Tensor>,device:&Device,linear:&Linear)->Result<(BTreeMap<String,Tensor>,Value)>{
    execute_range(gold,template,checkpoint,stage,forward,inputs,device,linear,0..48)
}

pub fn execute_range(gold:&Gold,template:&Value,checkpoint:&Checkpoint,stage:usize,forward:usize,
    inputs:BTreeMap<String,Tensor>,device:&Device,linear:&Linear,blocks:std::ops::Range<usize>)
    ->Result<(BTreeMap<String,Tensor>,Value)>{
    execute_range_with(gold,template,checkpoint,stage,forward,inputs,device,linear,blocks,None)
}

/// `execute_range` of a sequence-parallel local program (`sp::localize`), gathering attention
/// operands from the other GPU through `gather`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_range_with(gold:&Gold,template:&Value,checkpoint:&Checkpoint,stage:usize,forward:usize,
    mut inputs:BTreeMap<String,Tensor>,device:&Device,linear:&Linear,blocks:std::ops::Range<usize>,
    gather:Option<&dyn graph::SequenceGather>)->Result<(BTreeMap<String,Tensor>,Value)>{
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
    // Every block runs the same program: plan it once.
    let plan=graph::plan(template,false)?;
    for block in blocks{
        let setup=graph::host_profiling().then(std::time::Instant::now);
        let mut tensors=inputs.clone();
        for binding in bindings(template,checkpoint,&prefix,block)?.iter(){
            let value=if let Some(value)=device.cached_weight(&binding.cache_key){value}else{
                let expected=if block==0{Some(gold.entry(&binding.name)?.sha256.as_str())}else{None};
                let data=checkpoint.tensor(&binding.key,expected)?;
                let value=Tensor::allocate(binding.shape.clone(),contiguous_strides(&binding.shape),binding.dtype.into(),&device)?;
                upload_into(&value.storage,&data,&device)?;device.cache_weight(binding.cache_key.clone(),&value);value
            };
            if value.shape!=binding.shape || value.dtype!=binding.dtype{return Err("cached model parameter layout differs".into());}
            tensors.insert(binding.name.clone(),value);
        }
        // Validate branch-independent context/position/timestep identity before
        // reusing a block template. No later block activation is loaded. The profile does not change
        // while it is served: each (stage, forward, block) is validated once per thread.
        let validated=(gold.root.clone(),stage,forward,block);
        if !VALIDATED.with(|v|v.borrow().contains(&validated)){
            for name in inputs.keys().filter(|n|*n!=&video_key && *n!=&audio_key){
                let suffix=name.strip_prefix(&prefix).ok_or("template prefix")?;
                let first=gold.entry(&format!("block.{stage}.{forward}.0{suffix}"))?;
                let later=gold.entry(&format!("block.{stage}.{forward}.{block}{suffix}"))?;
                if first.sha256!=later.sha256 || first.shape!=later.shape || first.dtype!=later.dtype{
                    return Err(format!("block {block} changes prepared input {suffix}").into());
                }
            }
            VALIDATED.with(|v|v.borrow_mut().insert(validated));
        }
        if let Some(t)=setup{graph::host_record("backbone.setup",t);}
        let (values,_) = graph::execute_planned(&template,&plan,tensors,&device,&linear,gather)?;
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
    }
    // Blocks are stream-ordered on this device, so the host issues the next block while the GPU
    // still runs this one (freed activations are reused in stream order). One synchronization per
    // range: a two-GPU partition hands its outputs to the other device only once they are written.
    let synced=graph::host_profiling().then(std::time::Instant::now);
    device.sync()?;
    if let Some(t)=synced{graph::host_record("backbone.sync",t);}
    Ok((inputs,json!({"passed":checks.len()==count*2,"rung":"backbone","checks":checks,
        "device_uuid":device.uuid()?,"device_index":device.index})))
}
