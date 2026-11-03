//! Forward rung: modality input tensors → projections/AdaLN/RoPE → 48 AV
//! blocks → output heads. Gold activations are comparison targets only after
//! the initial forward inputs. This is not a raw-request inference entrypoint.
use crate::{backbone,cuda::Device,gold::{Gold,sha},graph,linear::Linear,
    tensor::{self,Tensor},weights::Checkpoint,Result};
use serde_json::{json,Value};
use std::collections::BTreeMap;

fn program(aux:&Gold,name:&str)->Result<Value>{
    let started=graph::host_profiling().then(std::time::Instant::now);
    if !name.bytes().all(|c|c.is_ascii_alphanumeric() || c==b'.' || c==b'-'){
        return Err("invalid component graph name".into());
    }
    let graph:Value=serde_json::from_slice(&std::fs::read(aux.root.join(format!("{name}.graph.json")))?)?;
    if graph["schema"]!="ltx25-aten-graph-v1"{return Err("component graph schema".into());}
    if let Some(t)=started{graph::host_record("forward.parse",t);}Ok(graph)
}

pub fn validate_identity(gold:&Gold,aux:&Gold,stage:usize)->Result<()>{
    if stage>1 || aux.identity["source_gold"].as_str()!=gold.root.to_str(){
        return Err("forward components do not belong to this admitted source".into());
    }
    for key in ["commit","tree_sha256","model_revision","weights","request"]{
        if aux.identity[key]!=gold.identity[key]{return Err(format!("component identity differs: {key}").into());}
    }
    Ok(())
}

pub struct Inputs {pub latent:Tensor,pub sigma:Tensor,pub timesteps:Tensor,pub positions:Option<Tensor>,pub keyframes:Option<Tensor>}
pub struct Output {pub outputs:Option<[Tensor;2]>,pub checks:Vec<Value>}

/// `LTX25_CUDA_GRAPHS=0` runs every sampling forward eagerly.
pub(crate) fn graphs_enabled()->bool{
    static ON:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
    *ON.get_or_init(||std::env::var("LTX25_CUDA_GRAPHS").as_deref()!=Ok("0"))
}
/// One sampling forward recorded as a CUDA graph (after an eager forward of the stage has loaded
/// its weights, constants, plans and kernels) and replayed for the stage's later steps. The graph
/// reads its inputs from staging tensors it owns: `run` copies each step's inputs there, launches
/// it and returns its output tensors, which the next launch overwrites.
pub struct CapturedForward {graph:crate::cuda::Graph,staging:[Inputs;2],contexts:Option<[Tensor;2]>,outputs:[Tensor;2]}
/// Captures kept for later requests of a resident worker: a stage's forward has the same layouts,
/// weights and constants for every request of a profile, so after the first request every step
/// replays. Keyed by device stream, weight-cache generation, profile root and stage; a capture
/// whose key no longer matches (other device, evicted weights) is dropped instead of reused.
type CaptureKey=(usize,u64,std::path::PathBuf,usize);
thread_local!{static CAPTURES:std::cell::RefCell<BTreeMap<CaptureKey,std::rc::Rc<CapturedForward>>>=
    const{std::cell::RefCell::new(BTreeMap::new())};}
fn capture_key(aux:&Gold,stage:usize,device:&Device)->CaptureKey{
    (device.stream as usize,device.weight_generation.get(),aux.root.clone(),stage)
}
pub(crate) fn kept_capture(aux:&Gold,stage:usize,device:&Device)->Option<std::rc::Rc<CapturedForward>>{
    let key=capture_key(aux,stage,device);
    CAPTURES.with(|c|{
        let mut c=c.borrow_mut();
        c.retain(|k,_|k.0!=key.0 || k.1==key.1);
        c.get(&key).cloned()
    })
}
pub(crate) fn keep_capture(aux:&Gold,stage:usize,device:&Device,value:std::rc::Rc<CapturedForward>){
    let key=capture_key(aux,stage,device);
    CAPTURES.with(|c|{c.borrow_mut().insert(key,value);});
}
fn staged(x:&Tensor,device:&Device)->Result<Tensor>{
    let value=Tensor::allocate(x.shape.clone(),x.strides.clone(),x.dtype.clone(),device)?;
    tensor::copy(x,&value,device.stream)?;Ok(value)
}
fn restage(x:&Tensor,staging:&Tensor,device:&Device)->Result<()>{
    if x.shape!=staging.shape || x.dtype!=staging.dtype{return Err("captured forward input layout differs".into());}
    tensor::copy(x,staging,device.stream)
}
impl CapturedForward {
    #[allow(clippy::too_many_arguments)]
    pub fn capture(gold:&Gold,aux:&Gold,checkpoint:&Checkpoint,stage:usize,forward:usize,dynamic:&[Inputs;2],
        contexts:Option<&[Tensor;2]>,device:&Device,linear:&Linear)->Result<Option<Self>>{
        if gold.verify{return Err("verified forwards are not captured".into());}
        let stage_inputs=|x:&Inputs|->Result<Inputs>{Ok(Inputs{latent:staged(&x.latent,device)?,sigma:staged(&x.sigma,device)?,
            timesteps:staged(&x.timesteps,device)?,positions:x.positions.as_ref().map(|t|staged(t,device)).transpose()?,
            keyframes:x.keyframes.as_ref().map(|t|staged(t,device)).transpose()?})};
        let staging=[stage_inputs(&dynamic[0])?,stage_inputs(&dynamic[1])?];
        let contexts=contexts.map(|c|->Result<[Tensor;2]>{Ok([staged(&c[0],device)?,staged(&c[1],device)?])}).transpose()?;
        let cached=device.cached_weight_tensors();
        let (output,graph)=device.capture(||run_with_contexts(gold,aux,checkpoint,stage,forward,Some(&staging),
            contexts.as_ref(),device,linear))?;
        if device.cached_weight_tensors()!=cached{return Err("a captured forward cached new weights".into());}
        let Some(outputs)=output.outputs else{return Ok(None)};
        Ok(Some(Self{graph,staging,contexts,outputs}))
    }
    pub fn run(&self,dynamic:&[Inputs;2],contexts:Option<&[Tensor;2]>,device:&Device)->Result<[Tensor;2]>{
        for (x,staging) in dynamic.iter().zip(&self.staging){
            restage(&x.latent,&staging.latent,device)?;restage(&x.sigma,&staging.sigma,device)?;
            restage(&x.timesteps,&staging.timesteps,device)?;
            match (&x.positions,&staging.positions){(Some(x),Some(s))=>restage(x,s,device)?,(None,None)=>{},
                _=>return Err("captured forward positions differ".into())}
            match (&x.keyframes,&staging.keyframes){(Some(x),Some(s))=>restage(x,s,device)?,(None,None)=>{},
                _=>return Err("captured forward keyframes differ".into())}
        }
        match (contexts,&self.contexts){
            (Some(c),Some(s))=>{restage(&c[0],&s[0],device)?;restage(&c[1],&s[1],device)?;}
            (None,None)=>{},_=>return Err("captured forward contexts differ".into()),
        }
        self.graph.launch()?;
        Ok(self.outputs.clone())
    }
}

pub fn run(gold:&Gold,aux:&Gold,checkpoint:&Checkpoint,stage:usize,forward:usize,
    dynamic:Option<&[Inputs;2]>,device:&Device,linear:&Linear)->Result<Output>{
    run_with_contexts(gold,aux,checkpoint,stage,forward,dynamic,None,device,linear)
}

fn component_weights(graph:&Value,checkpoint:&Checkpoint,prefix:&str,device:&Device,replacements:&mut BTreeMap<String,Tensor>)->Result<()>{
    let used=graph::external_names(graph)?;
    for name in graph["externals"].as_object().ok_or("component externals")?.keys(){
        if !used.contains(name){continue;}
        if let Some(suffix)=name.strip_prefix(&format!("{prefix}.weights.")){
            replacements.insert(name.clone(),crate::text::weight(checkpoint,&format!("model.diffusion_model.{suffix}"),device)?);
        }
    }
    Ok(())
}

pub fn run_with_contexts(gold:&Gold,aux:&Gold,checkpoint:&Checkpoint,stage:usize,forward:usize,
    dynamic:Option<&[Inputs;2]>,contexts:Option<&[Tensor;2]>,device:&Device,linear:&Linear)->Result<Output>{
    run_with_contexts_parallel(gold,aux,checkpoint,stage,forward,dynamic,contexts,device,linear,None)
}

pub fn run_with_contexts_parallel(gold:&Gold,aux:&Gold,checkpoint:&Checkpoint,stage:usize,forward:usize,
    dynamic:Option<&[Inputs;2]>,contexts:Option<&[Tensor;2]>,device:&Device,linear:&Linear,
    peer:Option<&crate::parallel::ParallelBlocks>)->Result<Output>{
    let parsed=graph::host_profiling().then(std::time::Instant::now);
    let template:Value=serde_json::from_slice(&std::fs::read(gold.root.join(format!("block.{stage}.0.0.graph.json")))?)?;
    if let Some(t)=parsed{graph::host_record("forward.parse_block",t);}
    let prefix=format!("block.{stage}.0.0");let mut inputs=BTreeMap::<String,Tensor>::new();let mut checks=Vec::new();
    for (modality,label) in ["video","audio"].into_iter().enumerate(){
        let name=format!("prep.{}",stage*2+modality);let graph=program(aux,&name)?;
        let mut replacements=BTreeMap::new();
        component_weights(&graph,checkpoint,&name,device,&mut replacements)?;
        if let Some(contexts)=contexts{
            for argument in 0..2{
                replacements.insert(format!("{name}.args.{argument}.context"),contexts[if argument==0{modality}else{1-modality}].clone());
            }
        }
        if let Some(dynamic)=dynamic{
            for argument in 0..2{
                let input=&dynamic[if argument==0{modality}else{1-modality}];
                for (field,value) in [("latent",&input.latent),("sigma",&input.sigma),("timesteps",&input.timesteps)]{
                    replacements.insert(format!("{name}.args.{argument}.{field}"),value.clone());
                }
                if let Some(value)=&input.positions{replacements.insert(format!("{name}.args.{argument}.positions"),value.clone());}
                if let Some(value)=&input.keyframes{replacements.insert(format!("{name}.args.{argument}.keyframes_mask"),value.clone());}
            }
        }
        let external=graph::load_externals_replacing(aux,&graph,&device,&replacements)?;
        if stage==0 && forward==0 && modality==0{native_core_infra::execution_start::arm();}
        let (values,_)=graph::execute(&graph,external,&device,&linear,false)?;
        let field_prefix=format!("{prefix}.kwargs.{label}.");
        for (key,spec) in template["externals"].as_object().ok_or("block externals")?{
            let Some(field)=key.strip_prefix(&field_prefix)else{continue;};
            let value=if field=="cross_attn_perturbation_mask"{
                // The admitted official branch passes perturbations=None;
                // BlockPerturbationsProcessor creates an all-keep mask.
                let shape=spec["shape"].as_array().ok_or("mask shape")?.iter()
                    .map(|v|v.as_u64().map(|n|n as usize).ok_or("mask extent")).collect::<std::result::Result<Vec<_>,_>>()?;
                let value=Tensor::contiguous_shape(shape,spec["dtype"].as_str().ok_or("mask dtype")?.into(),&device)?;
                tensor::fill(&value,1.,0.,device.stream)?;value
            }else{
                let id=graph["output_paths"][format!(".{field}")].as_str()
                    .ok_or_else(||format!("preprocessor has no field {field}"))?;
                values.get(id).ok_or("undefined preprocessor output")?.clone()
            };
            let reference=key.replacen(&prefix,&format!("block.{stage}.{forward}.0"),1);
            let actual=if gold.verify{sha(&value.download(&device)?)}else{String::new()};let expected=&gold.entry(&reference)?.sha256;
            let passed=!gold.verify || actual==*expected;
            checks.push(json!({"phase":"prepared","tensor":key,"passed":passed,"sha256":actual,"expected_sha256":expected}));
            if !passed{return Ok(Output{outputs:None,checks});}
            inputs.insert(key.clone(),value);
        }
    }
    let (inputs,body)=if peer.is_some(){
        backbone::execute_range(gold,&template,checkpoint,stage,forward,inputs,&device,&linear,0..24)?
    }else{backbone::execute(gold,&template,checkpoint,stage,forward,inputs,&device,&linear)?};
    checks.extend(body["checks"].as_array().ok_or("backbone report")?.iter().cloned());
    if body["passed"]!=true{return Ok(Output{outputs:None,checks});}
    let inputs=if let Some(peer)=peer{
        let (inputs,body)=peer.forward(gold,stage,forward,inputs,device)?;
        checks.extend(body["checks"].as_array().ok_or("secondary backbone report")?.iter().cloned());
        if body["passed"]!=true{return Ok(Output{outputs:None,checks});}inputs
    }else{inputs};
    let mut outputs=Vec::new();
    for (modality,label) in ["video","audio"].into_iter().enumerate(){
        let name=format!("head.{}",stage*2+modality);let graph=program(aux,&name)?;
        let mut replacements=BTreeMap::from([
            (format!("{name}.args.3"),inputs[&format!("{prefix}.kwargs.{label}.x")].clone()),
            (format!("{name}.args.4"),inputs[&format!("{prefix}.kwargs.{label}.embedded_timestep")].clone()),
        ]);
        component_weights(&graph,checkpoint,&name,device,&mut replacements)?;
        for key in replacements.keys(){if graph["externals"].get(key).is_none(){
            return Err(format!("head does not consume expected native predecessor {key}").into());}}
        let external=graph::load_externals_replacing(aux,&graph,&device,&replacements)?;
        let (values,_)=graph::execute(&graph,external,&device,&linear,false)?;
        let id=graph["output_paths"][""].as_str().ok_or("head output path")?;
        let value=values.get(id).ok_or("head output")?;
        let actual=if gold.verify{sha(&value.download(&device)?)}else{String::new()};
        let key=format!("forward.{stage}.{forward}.out.{modality}");let expected=&gold.entry(&key)?.sha256;
        let passed=!gold.verify || actual==*expected;
        checks.push(json!({"phase":"output","tensor":key,"passed":passed,"sha256":actual,"expected_sha256":expected}));
        if !passed{return Ok(Output{outputs:None,checks});}
        outputs.push(value.clone());
    }
    Ok(Output{outputs:Some(outputs.try_into().map_err(|_|"expected both modality outputs")?),checks})
}
