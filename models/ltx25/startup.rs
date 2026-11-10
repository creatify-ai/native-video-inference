//! LTX declares tensor dependencies; core_infra owns reads, verification,
//! retained RAM leases, CUDA context binding and staged uploads.
use crate::{cuda::{Buffer,Device},tensor::{self,Tensor},weights::Checkpoint,Result};
use native_core_infra::{checkpoint::{Declaration,Dtype},model_load::{self,Loaded,Options,Request,Components}};
use std::rc::Rc;

/// The dual-GPU owner retains its existing structured CPU preparation path.
pub(crate) fn prepare_cpu(declaration:&Declaration)->Result<native_core_infra::startup_policy::CpuPreparation<crate::frontend::GemmaTokenizer>>{
    let manifest=std::sync::Arc::new(declaration.manifest.clone());let roots=declaration.roots.clone();
    Ok(native_core_infra::startup_policy::CpuPreparation::start("model-tokenizer",move||
        tokenizer_parts(manifest,roots).map_err(|error|error.to_string()))?)
}

pub(crate) fn tokenizer(declaration:&Declaration)->Result<crate::frontend::GemmaTokenizer>{
    let (manifest,roots)=native_core_infra::startup_trace::measure("tokenizer_prepare","clone metadata",0,||
        (std::sync::Arc::new(declaration.manifest.clone()),declaration.roots.clone()));
    tokenizer_parts(manifest,roots)
}
fn tokenizer_parts(manifest:std::sync::Arc<native_core_infra::checkpoint::Manifest>,
                   roots:std::collections::HashMap<String,std::path::PathBuf>)->Result<crate::frontend::GemmaTokenizer>{
    let entry=manifest.tensor("text.tokenizer_json")?;let file=&manifest.files[&entry.file];
    let path=roots.get(&file.root).ok_or("tokenizer checkpoint root")?.join(&file.path);
    let mut checkpoint=native_core_infra::startup_trace::measure("tokenizer_prepare","open checkpoint",0,||Checkpoint::open(path))?;
    native_core_infra::startup_trace::measure("tokenizer_prepare","admit checkpoint",0,||
        checkpoint.admit("text",&file.sha256,manifest.clone(),&roots))?;
    crate::frontend::GemmaTokenizer::from_checkpoint(&checkpoint)
}

pub(crate) fn stages(declaration:&Declaration)->Result<Vec<Vec<Request>>> {
    let mut stages:Vec<Vec<Request>>=(0..5).map(|_|Vec::new()).collect();
    for (name,t) in &declaration.manifest.tensors {
        if !matches!(t.dtype,Dtype::BF16|Dtype::F32){continue;}
        let stage=if name.starts_with("text."){0}
            else if name.starts_with("dit.model.diffusion_model.video_embeddings_connector.")
                || name.starts_with("dit.model.diffusion_model.audio_embeddings_connector."){1}
            else if name.starts_with("dit."){2}
            else if name.starts_with("vae."){3}
            else if name.starts_with("upscale."){4}
            else{return Err(format!("undeclared LTX startup component: {name}").into());};
        stages[stage].push(Request{name:name.clone(),dtype:t.dtype});
    }
    Ok(stages)
}
/// GPU 0 loads the whole model with one GPU or two (sequence parallel: every block runs on both).
fn stages_world(declaration:&Declaration,world:usize)->Result<Vec<Vec<Request>>>{
    if !matches!(world,1|2){return Err("LTX supports one or two physical GPUs".into());}
    stages(declaration)
}
pub(crate) fn start_world(declaration:&Declaration,world:usize)->Result<Components> {
    Ok(model_load::start_components(&declaration.manifest,&declaration.roots,stages_world(declaration,world)?,Options::for_declaration(declaration)?)?)
}

pub(crate) fn prepare_host(declaration:&Declaration)->Result<model_load::HostPreparation> {
    prepare_host_world(declaration,1)
}
pub(crate) fn prepare_host_world(declaration:&Declaration,world:usize)->Result<model_load::HostPreparation> {
    let mut options=Options::for_declaration(declaration)?;
    // Same resident host budget as the original LTX session. Admission is
    // shared; no checkpoint layout, dtype, CPU count or reader tuning changes.
    options.retained_budget=80usize<<30;
    Ok(model_load::HostPreparation::start(declaration.manifest.clone(),declaration.roots.clone(),stages_world(declaration,world)?,options)?)
}

pub(crate) fn install(loaded:Loaded,checkpoints:&[(&str,Rc<Checkpoint>)],device:&Device)->Result<()> {
    // Validate every view before transferring ownership out of the framework.
    for (name,view) in &loaded.tensors {
        let (prefix,key)=name.split_once('.').ok_or("LTX manifest tensor prefix")?;
        let checkpoint=&checkpoints.iter().find(|(p,_)|*p==prefix).ok_or("LTX component")?.1;
        let (shape,dtype)=checkpoint.weight_layout(key)?;
        let expected=match view.dtype {Dtype::BF16=>"torch.bfloat16",Dtype::F32=>"torch.float32",_=>return Err("LTX weight dtype".into())};
        if shape!=view.shape || dtype!=expected{return Err("LTX framework view layout differs".into());}
    }
    let (pointer,views,host,stats)=unsafe{loaded.into_allocation()};
    let storage=Rc::new(unsafe{Buffer::from_allocation(pointer,stats.device_bytes)});
    for (name,view) in views {
        let (prefix,key)=name.split_once('.').unwrap();
        let checkpoint=&checkpoints.iter().find(|(p,_)|*p==prefix).unwrap().1;
        let (shape,dtype)=checkpoint.weight_layout(key)?;
        let value=Tensor{storage:storage.clone(),offset:(view.ptr-pointer as u64) as usize,
            strides:tensor::contiguous_strides(&shape),shape,dtype,reference_offset:0};
        device.cache_weight(format!("{}\0{key}",checkpoint.path.display()),&value);
        if let Some(host)=&host{checkpoint.retain(key,&name,host.clone());}
    }
    eprintln!("{}",serde_json::json!({"event":"framework-stage-installed","tensors":stats.tensors,
        "device_bytes":stats.device_bytes,"read_bytes":stats.read_bytes,"seconds":stats.seconds}));
    Ok(())
}

pub(crate) fn install_host(host:std::sync::Arc<model_load::HostWeights>,checkpoints:&[(&str,Rc<Checkpoint>)])->Result<()> {
    for name in host.names() {
        let (prefix,key)=name.split_once('.').ok_or("LTX manifest prefix")?;
        let checkpoint=&checkpoints.iter().find(|(p,_)|*p==prefix).ok_or("LTX component")?.1;
        checkpoint.weight_layout(key)?;checkpoint.retain(key,name,host.clone());
    }
    Ok(())
}
