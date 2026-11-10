//! Ancestral Euler rungs. A stage starts from its admitted front-end tensors;
//! every subsequent latent, velocity, x0 and noise draw is computed natively.
use crate::{cuda::{Device,upload_into},forward::{self,Inputs},gold::{Gold,sha},linear::Linear,
    ops,rng::CudaGenerator,tensor::{self,Tensor},weights::Checkpoint,Result};
use serde_json::{json,Value};

fn shape(value:&Value)->Result<Vec<usize>>{
    value.as_array().ok_or("tensor shape")?.iter().map(|v|v.as_u64().map(|n|n as usize)
        .ok_or_else(||"invalid extent".into())).collect()
}
fn load(gold:&Gold,name:&str,spec:Option<&Value>,device:&Device)->Result<Tensor>{
    let entry=gold.entry(name)?;
    let value=Tensor::contiguous_shape(entry.shape.clone(),entry.dtype.clone(),device)?;
    let data=gold.asset(name)?;upload_into(&value.storage,&data,device)?;
    if let Some(spec)=spec{
        let restored=Tensor::allocate(shape(&spec["shape"])?,shape(&spec["stride"])?,entry.dtype.clone(),device)?;
        tensor::copy(&value,&restored,device.stream)?;Ok(restored)
    }else{Ok(value)}
}
fn cast(x:&Tensor,dtype:&str,device:&Device)->Result<Tensor>{
    let value=Tensor::contiguous_shape(x.shape.clone(),dtype.into(),device)?;
    tensor::copy(x,&value,device.stream)?;Ok(value)
}
fn binary(op:i32,x:&Tensor,y:Option<&Tensor>,scalar:f32,dtype:&str,device:&Device)->Result<Tensor>{
    let value=Tensor::contiguous_shape(x.shape.clone(),dtype.into(),device)?;
    tensor::elementwise(op,x,y,None,&value,scalar,1.,device.stream)?;Ok(value)
}
fn post_process(x:&Tensor,mask:&Tensor,clean:&Tensor,device:&Device)->Result<Tensor>{
    if x.dtype!="torch.float32"{return Err("sampling post-process expects FP32".into());}
    let masked=binary(21,x,Some(mask),0.,"torch.float32",device)?;
    let inverse=binary(24,mask,None,1.,&mask.dtype,device)?;
    let clean=cast(clean,"torch.float32",device)?;
    let conditioned=binary(21,&clean,Some(&inverse),0.,"torch.float32",device)?;
    binary(20,&masked,Some(&conditioned),0.,"torch.float32",device)
}
fn comparison(gold:&Gold,name:&str,value:&Tensor,device:&Device,checks:&mut Vec<Value>)->Result<bool>{
    if !gold.verify{return Ok(true);}
    let entry=gold.entry(name)?;
    let actual=sha(&value.download(device)?);
    let passed=actual==entry.sha256 && value.dtype==entry.dtype && value.shape==entry.shape;
    checks.push(json!({"tensor":name,"passed":passed,"sha256":actual,"expected_sha256":entry.sha256}));
    Ok(passed)
}

pub struct State {pub latent:Tensor,pub mask:Tensor,pub clean:Tensor,pub positions:Option<Tensor>,pub keyframes:Option<Tensor>}
pub struct Outcome {pub states:Vec<Tensor>,pub report:Value}

pub fn run(gold:&Gold,aux:&Gold,checkpoint:&Checkpoint,stage:usize,single_step:bool,
    initial:Option<&[State;2]>,contexts:Option<&[Tensor;2]>,device:&Device,linear:&Linear)->Result<Outcome>{
    run_parallel(gold,aux,checkpoint,stage,single_step,initial,contexts,device,linear,None)
}

pub fn run_parallel(gold:&Gold,aux:&Gold,checkpoint:&Checkpoint,stage:usize,single_step:bool,
    initial:Option<&[State;2]>,contexts:Option<&[Tensor;2]>,device:&Device,linear:&Linear,
    peer:Option<&crate::parallel::ParallelBlocks>)->Result<Outcome>{
    forward::validate_identity(gold,aux,stage)?;
    let step_base=if stage==0{0}else{14};let global_base=if stage==0{0}else{8};
    let sigma_name=format!("op.ancestral.{step_base}.sigmas");
    let sigma_raw=gold.asset(&sigma_name)?;
    if sigma_raw.len()%4!=0{return Err("FP32 sigma schedule".into());}
    let schedule=sigma_raw.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    let steps=if stage==0{8}else{3};
    if schedule.len()!=steps+1 || schedule[steps]!=0.{return Err("only the frozen official schedules are ported".into());}
    let sigma_buffer=crate::cuda::Buffer::upload(&sigma_raw,&device)?;
    let seed=gold.identity["request"]["seed"].as_u64().ok_or("request seed")?
        .checked_add(if stage==0{10000}else{20000}).ok_or("seed exceeds the admitted unsigned range")?;
    let mut generator=CudaGenerator::new(seed)?;
    let draws=gold.metadata["fixtures"].as_array().ok_or("RNG fixtures")?.iter().filter_map(|fixture|{
        let name=fixture["name"].as_str()?;
        (fixture["kind"]=="rng" && gold.metadata[name]["seed"].as_u64()==Some(seed)).then(||name.to_owned())
    }).collect::<Vec<_>>();
    let draws=if gold.verify{draws}else{vec![String::new();2*(steps-1)]};
    if draws.len()!=2*(steps-1){return Err("ancestral RNG draw count differs from frozen profile".into());}
    let mut states=Vec::new();let mut masks=Vec::new();let mut cleans=Vec::new();
    for modality in 0..2{
        let name=format!("prep.{}",stage*2+modality);
        let graph:Value=serde_json::from_slice(&std::fs::read(aux.root.join(format!("{name}.graph.json")))?)?;
        let key=format!("{name}.args.0.latent");
        if let Some(initial)=initial{
            let native=&initial[modality];let spec=&graph["externals"][&key];
            let restored=Tensor::allocate(shape(&spec["shape"])?,shape(&spec["stride"])?,native.latent.dtype.clone(),device)?;
            tensor::copy(&native.latent,&restored,device.stream)?;states.push(restored);
            masks.push(native.mask.clone());cleans.push(native.clean.clone());
        }else{
            states.push(load(aux,&key,Some(&graph["externals"][&key]),&device)?);
            let noiser=format!("noiser.{}",stage*2+modality);
            masks.push(load(gold,&format!("{noiser}.out.denoise_mask"),None,&device)?);
            cleans.push(load(gold,&format!("{noiser}.out.clean_latent"),None,&device)?);
        }
    }
    let mut checks=Vec::new();let count=if single_step{1}else{
        native_core_infra::switch_benchmark::steps(steps,gold.verify)?
    };let mut draw=0;let mut unpatchified=Vec::new();
    let passed=(||->Result<bool>{
        for step in 0..count{
            let sigma=Tensor::contiguous_shape(vec![1],"torch.float32".into(),&device)?;
            tensor::fill(&sigma,schedule[step],0.,device.stream)?;
            let make_input=|modality:usize|->Result<Inputs>{Ok(Inputs{
                latent:states[modality].clone(),sigma:sigma.clone(),
                timesteps:binary(21,&masks[modality],Some(&sigma),0.,"torch.float32",&device)?,
                positions:initial.and_then(|s|s[modality].positions.clone()),
                keyframes:initial.and_then(|s|s[modality].keyframes.clone()),
            })};
            let inputs=[make_input(0)?,make_input(1)?];
            let result=forward::run_with_contexts_parallel(gold,aux,checkpoint,stage,step,Some(&inputs),contexts,device,linear,peer)?;
            checks.extend(result.checks);
            let Some(velocity)=result.outputs else{return Ok(false);};
            for modality in 0..2{
                // X0Model: materialize multiply and subtract in FP32, then
                // round back to BF16 before the sampler's FP32 mask blend.
                let sample=cast(&states[modality],"torch.float32",&device)?;
                let v=cast(&velocity[modality],"torch.float32",&device)?;
                let scaled=binary(21,&v,Some(&inputs[modality].timesteps),0.,"torch.float32",&device)?;
                let denoised=binary(22,&sample,Some(&scaled),0.,"torch.float32",&device)?;
                let x0=cast(&denoised,"torch.bfloat16",&device)?;
                if !comparison(gold,&format!("x0.{}.out.{modality}",global_base+step),&x0,&device,&mut checks)?{
                    return Ok(false);
                }
                let denoised=post_process(&cast(&x0,"torch.float32",&device)?,&masks[modality],&cleans[modality],&device)?;
                let next=if schedule[step+1]==0.{denoised}else{
                    let fixture=&draws[draw];draw+=1;
                    let metadata=&gold.metadata[fixture];
                    if gold.verify && metadata["offset_before"].as_u64()!=Some(generator.offset){return Err("Philox draw order differs".into());}
                    let noise=Tensor::contiguous_shape(states[modality].shape.clone(),"torch.bfloat16".into(),&device)?;
                    generator.randn(noise.ptr(),noise.elements()?,true,device.stream)?;
                    if gold.verify && metadata["offset_after"].as_u64()!=Some(generator.offset){return Err("Philox offset differs".into());}
                    if gold.verify && !comparison(gold,&format!("{fixture}.out"),&noise,&device,&mut checks)?{return Ok(false);}
                    let name=format!("op.ancestral.{}",step_base+step*2+modality);
                    if gold.scalar(&format!("{name}.eta"))?!=1. || gold.scalar(&format!("{name}.s_noise"))?!=1.{
                        return Err("only official eta=1, s_noise=1 is ported".into());
                    }
                    if !comparison(gold,&format!("{name}.sample"),&sample,&device,&mut checks)? ||
                        !comparison(gold,&format!("{name}.denoised_sample"),&denoised,&device,&mut checks)?{
                        return Ok(false);
                    }
                    let output=Tensor::contiguous_shape(sample.shape.clone(),"torch.float32".into(),&device)?;
                    ops::ancestral(sample.ptr(),denoised.ptr(),noise.ptr(),output.ptr(),output.elements()?,
                        sigma_buffer.dev(),step,1.,1.,true,device.stream)?;
                    if !comparison(gold,&format!("{name}.output"),&output,&device,&mut checks)?{return Ok(false);}
                    post_process(&output,&masks[modality],&cleans[modality],&device)?
                };
                let updated=Tensor::allocate(states[modality].shape.clone(),states[modality].strides.clone(),
                    "torch.bfloat16".into(),&device)?;
                tensor::copy(&next,&updated,device.stream)?;states[modality]=updated;
                if step+1<steps{
                    let name=format!("forward.{stage}.{}.args.{modality}.latent",step+1);
                    if !comparison(gold,&name,&states[modality],&device,&mut checks)?{return Ok(false);}
                }
            }
            eprintln!("LTX25_NATIVE_STEP stage={stage} step={step} passed=true");
        }
        if !single_step{
            for modality in 0..2{
                let label=if modality==0{"video"}else{"audio"};
                let name=format!("stage.{stage}.out.{label}.latent");
                let target=&gold.entry(&name)?.shape;let mut restored=states[modality].clone();
                if modality==0 && target.len()==5{
                    if target[1]!=128 || target[2]*target[3]*target[4]!=restored.shape[1]{return Err("video unpatchify extent".into());}
                    let s=&states[modality].strides;
                    restored.strides=vec![s[0],s[2],s[1]*target[3]*target[4],s[1]*target[4],s[1]];
                }else if modality==1 && target.len()==4{
                    if target[1]*target[3]!=128 || target[2]!=restored.shape[1]{return Err("audio unpatchify extent".into());}
                    let s=&states[modality].strides;restored.strides=vec![s[0],s[2]*target[3],s[1],s[2]];
                }else{return Err("unported stage output shape".into());}
                restored.shape=target.clone();
                if !comparison(gold,&name,&restored,&device,&mut checks)?{return Ok(false);}
                unpatchified.push(restored);
            }
        }
        Ok(true)
    })()?;
    Ok(Outcome{states:unpatchified,report:json!({"passed":passed,"rung":if single_step{"step"}else{"segment"},"stage":stage,
        "steps":count,"checks":checks})})
}
