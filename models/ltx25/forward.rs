//! Forward rung: modality input tensors → projections/AdaLN/RoPE → 48 AV
//! blocks → output heads. Gold activations are comparison targets only after
//! the initial forward inputs. This is not a raw-request inference entrypoint.
use crate::{backbone,cuda::Device,gold::{Gold,sha},graph,linear::Linear,
    tensor::{self,Tensor},weights::Checkpoint,Result};
use serde_json::{json,Value};
use std::collections::BTreeMap;

thread_local!{static PARSED:std::cell::RefCell<BTreeMap<std::path::PathBuf,std::rc::Rc<Value>>>=
    const{std::cell::RefCell::new(BTreeMap::new())};}
/// A profile's program file, parsed once per thread: profile files do not change while a process
/// serves them (every forward of every request reads the same ones).
fn parsed_program(path:std::path::PathBuf)->Result<std::rc::Rc<Value>>{
    if let Some(value)=PARSED.with(|c|c.borrow().get(&path).cloned()){return Ok(value);}
    let value=std::rc::Rc::new(serde_json::from_slice::<Value>(&std::fs::read(&path)?)?);
    PARSED.with(|c|c.borrow_mut().insert(path,value.clone()));Ok(value)
}
fn program(aux:&Gold,name:&str)->Result<std::rc::Rc<Value>>{
    let started=graph::host_profiling().then(std::time::Instant::now);
    if !name.bytes().all(|c|c.is_ascii_alphanumeric() || c==b'.' || c==b'-'){
        return Err("invalid component graph name".into());
    }
    let graph=parsed_program(aux.root.join(format!("{name}.graph.json")))?;
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
    let template=parsed_program(gold.root.join(format!("block.{stage}.0.0.graph.json")))?;
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
    let (inputs,body)=if let Some(peer)=peer{
        peer.forward(gold,&template,checkpoint,stage,forward,inputs,device,linear)?
    }else{backbone::execute(gold,&template,checkpoint,stage,forward,inputs,&device,&linear)?};
    checks.extend(body["checks"].as_array().ok_or("backbone report")?.iter().cloned());
    if body["passed"]!=true{return Ok(Output{outputs:None,checks});}
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
