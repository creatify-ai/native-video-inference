//! Layer dataflow executed by Rust/CUDA only. No ATen or Python runtime.
//! Unknown operators/branches fail before their output can be accepted.
use crate::{cuda::Device, gold::{Gold, sha}, linear::Linear, sdpa,
    tensor::{self, Tensor}, Result};
use serde_json::{json, Value};
use std::{collections::{BTreeMap,BTreeSet}, path::Path};

/// `LTX25_HOST_PROFILE=1`: host seconds spent per graph operator and per named section, for
/// finding interpreter overhead (CPU sampling is unavailable in some containers).
pub(crate) fn host_profiling()->bool{
    static ON:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
    *ON.get_or_init(||std::env::var("LTX25_HOST_PROFILE").as_deref()==Ok("1"))
}
thread_local!{static HOST:std::cell::RefCell<BTreeMap<String,(u64,f64)>>=const{std::cell::RefCell::new(BTreeMap::new())};}
pub(crate) fn host_record(name:&str,started:std::time::Instant){
    let seconds=started.elapsed().as_secs_f64();
    HOST.with(|h|{let mut h=h.borrow_mut();
        if let Some(entry)=h.get_mut(name){entry.0+=1;entry.1+=seconds;}else{h.insert(name.into(),(1,seconds));}});
}
/// The recorded host profile, most expensive first, and resets it.
pub fn host_profile_take()->Value{
    let mut rows:Vec<(String,(u64,f64))>=HOST.with(|h|std::mem::take(&mut *h.borrow_mut())).into_iter().collect();
    rows.sort_by(|a,b|b.1.1.total_cmp(&a.1.1));
    Value::Array(rows.into_iter().map(|(name,(count,seconds))|json!({"name":name,"count":count,"seconds":seconds})).collect())
}

fn usize_list(value:&Value)->Result<Vec<usize>> {
    value.as_array().ok_or("expected shape array")?.iter()
        .map(|v|v.as_u64().map(|n|n as usize).ok_or_else(||"invalid dimension".into())).collect()
}
fn name(value:&Value)->Result<&str>{value["tensor"].as_str().ok_or_else(||"expected tensor operand".into())}
fn get<'a>(tensors:&'a BTreeMap<String,Tensor>,value:&Value)->Result<&'a Tensor>{
    tensors.get(name(value)?).ok_or_else(||format!("undefined tensor {}",value).into())
}
fn alloc(spec:&Value,device:&Device)->Result<Tensor>{
    let mut value=Tensor::allocate(usize_list(&spec["shape"])?,usize_list(&spec["stride"])?,
        spec["dtype"].as_str().ok_or("output dtype")?.into(),device)?;
    value.reference_offset=spec["storage_offset"].as_u64().unwrap_or(0) as usize;
    Ok(value)
}
fn specs<'a>(value:&'a Value,out:&mut Vec<&'a Value>){
    if value.get("tensor").is_some(){out.push(value);}
    else if let Some(values)=value.as_array(){for value in values{specs(value,out);}}
}

/// Model metadata only: build the same cuDNN plans on the serving thread
/// while the framework uploads weights. No activation or weight is read, and
/// no graph is executed. CUTLASS attention has no cuDNN plan to prepare.
pub(crate) fn prepare_attention(graph:&Value)->Result<usize>{
    let mut layouts:BTreeMap<String,&Value>=graph["externals"].as_object().ok_or("graph externals")?
        .iter().map(|(name,spec)|(name.clone(),spec)).collect();
    let mut prepared=0;
    for node in graph["nodes"].as_array().ok_or("graph nodes")?{
        let op=node["op"].as_str().ok_or("operator name")?;
        if matches!(op,"aten._scaled_dot_product_cudnn_attention.default"|"aten.scaled_dot_product_attention.default")
            && node["sdpa_backend"].as_i64()!=Some(2){
            let args=node["args"].as_array().ok_or("operator args")?;
            let view=|operand:&Value|->Result<sdpa::View>{
                let spec=*layouts.get(name(operand)?).ok_or("undefined attention metadata")?;
                if spec["dtype"]!="torch.bfloat16"{return Err("attention preparation dtype outside port".into());}
                let dims=usize_list(&spec["shape"])?.into_iter().map(i64::try_from).collect::<std::result::Result<Vec<_>,_>>()?;
                let strides=usize_list(&spec["stride"])?.into_iter().map(i64::try_from).collect::<std::result::Result<Vec<_>,_>>()?;
                Ok(sdpa::View{dims:dims.try_into().map_err(|_|"attention metadata rank")?,
                    strides:strides.try_into().map_err(|_|"attention metadata rank")?,offset_bytes:0})
            };
            let (q,k,v)=(view(&args[0])?,view(&args[1])?,view(&args[2])?);
            let mut mask=args.get(3).filter(|v|!v.is_null()).or_else(||node["kwargs"].get("attn_mask").filter(|v|!v.is_null()))
                .map(view).transpose()?;
            if let Some(mask)=&mut mask{
                for (axis,extent) in [(0,q.dims[0]),(2,q.dims[2]),(3,k.dims[2])]{
                    if mask.dims[axis]!=extent{
                        if mask.dims[axis]!=1{return Err("attention preparation bias is not broadcastable".into());}
                        mask.dims[axis]=extent;mask.strides[axis]=0;
                    }
                }
            }
            sdpa::prepare(sdpa::Io::Bf16,&q,&k,&v,mask.as_ref())?;prepared+=1;
        }
        let mut outputs=Vec::new();specs(&node["out"],&mut outputs);
        for spec in outputs{layouts.insert(name(spec)?.into(),spec);}
    }
    Ok(prepared)
}
fn alias(source:&Tensor,spec:&Value)->Result<Tensor>{
    let mut result=source.clone();
    result.shape=usize_list(&spec["shape"])?;result.strides=usize_list(&spec["stride"])?;
    let reference=spec["storage_offset"].as_u64().ok_or("view offset")? as usize;
    let width=tensor::dtype(&result.dtype)?.1;
    let offset=result.offset as i128+(reference as i128-source.reference_offset as i128)*width as i128;
    if offset<0 || offset as usize>result.storage.bytes(){return Err("view escapes source allocation".into());}
    result.offset=offset as usize;result.reference_offset=reference;Ok(result)
}
// Consecutive logical dimensions can be coalesced only when their physical
// addresses also form one run. Equal runs mean reshape can preserve storage.
fn storage_runs(shape:&[usize],strides:&[usize])->Vec<(usize,usize)>{
    let mut runs:Vec<(usize,usize)>=Vec::new();
    for (&size,&stride) in shape.iter().zip(strides).rev(){
        if size==1{continue;}
        if let Some((count,inner))=runs.last_mut(){
            if stride==*count * *inner{*count*=size;continue;}
        }
        runs.push((size,stride));
    }
    runs
}

pub fn load_externals(gold:&Gold,graph:&Value,device:&Device)->Result<BTreeMap<String,Tensor>>{
    load_externals_replacing(gold,graph,device,&BTreeMap::new())
}

pub fn external_names(graph:&Value)->Result<BTreeSet<String>>{
    fn visit(value:&Value,used:&mut BTreeSet<String>){
        match value{
            Value::Object(fields)=>{
                if let Some(name)=fields.get("tensor").and_then(Value::as_str){used.insert(name.into());}
                for value in fields.values(){visit(value,used);}
            }
            Value::Array(items)=>for value in items{visit(value,used);},_=>{}
        }
    }
    let mut used=BTreeSet::new();
    for node in graph["nodes"].as_array().ok_or("graph nodes")?{
        visit(&node["args"],&mut used);visit(&node["kwargs"],&mut used);
    }
    for output in graph["outputs"].as_array().ok_or("graph outputs")?{
        used.insert(output.as_str().ok_or("output name")?.into());
    }
    let declared=graph["externals"].as_object().ok_or("graph externals")?;
    used.retain(|name|declared.contains_key(name));
    Ok(used)
}

pub fn load_externals_replacing(gold:&Gold,graph:&Value,device:&Device,
    replacements:&BTreeMap<String,Tensor>)->Result<BTreeMap<String,Tensor>>{
    let _trace=native_core_infra::startup_trace::Span::new("graph_bind",gold.root.to_string_lossy(),0);
    let mut tensors=BTreeMap::<String,Tensor>::new();
    let used=external_names(graph)?;
    for (key,spec) in graph["externals"].as_object().ok_or("missing graph externals")? {
        if !used.contains(key){continue;}
        if let Some(value)=replacements.get(key){
            if json!(value.shape)!=spec["shape"] || value.dtype!=spec["dtype"].as_str().ok_or("replacement dtype")?{
                return Err(format!("replacement layout differs for {key}").into());
            }
            // Composition must not even read the recorded activation being
            // replaced. Only the native predecessor supplies these bytes.
            let _materialize=native_core_infra::startup_trace::Span::new("external_materialize",key,
                value.elements()?*tensor::dtype(&value.dtype)?.1);
            let restored=alloc(spec,device)?;
            tensor::copy(value,&restored,device.stream)?;
            tensors.insert(key.clone(),restored);
            continue;
        }
        let entry=gold.entry(key)?;
        // Serving profiles contain SHA-pinned model constants. Captured
        // activations have no file and request inputs take the replacement
        // branch above, so neither can enter this immutable weight cache.
        let cache_key=if !gold.verify && entry.file.is_some(){Some(format!("immutable-graph\0{}\0{key}\0{}\0{}",
            gold.root.display(),entry.sha256,serde_json::to_string(spec)?))}else{None};
        if let Some(value)=cache_key.as_ref().and_then(|key|device.cached_weight(key)){
            tensors.insert(key.clone(),value);continue;
        }
        let raw=Tensor::contiguous_shape(entry.shape.clone(),entry.dtype.clone(),&device)?;
        let data=gold.asset(key)?;
        crate::cuda::upload_into(&raw.storage,&data,&device)?;
        let restored=alloc(spec,&device)?;
        tensor::copy(&raw,&restored,device.stream)?;
        if let Some(cache_key)=cache_key{
            device.cache_weight(cache_key,&restored);device.record_asset_upload(data.len());
        }
        tensors.insert(key.clone(),restored);
    }
    Ok(tensors)
}

/// A run of consecutive eager elementwise nodes over one output shape, executed as one
/// `tensor::fused` kernel at its last node. Values used only inside the run stay in registers
/// (rounded to their eager dtype); `outputs` are the values used after it.
struct FusedRun {end:usize,inputs:Vec<String>,outputs:Vec<(Value,i32)>,ops:Vec<tensor::FuseOp>}
/// On by default; `LTX25_FUSE=0` runs every node as its own kernel. Each run is compiled once with
/// NVRTC (`LTX25_KERNEL_CACHE` keeps the compiled kernels across processes).
fn fusion_enabled()->bool{
    static ON:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
    *ON.get_or_init(||std::env::var("LTX25_FUSE").as_deref()!=Ok("0"))
}
const ALIAS_OPS:&[&str]=&["aten.view.default","aten._unsafe_view.default","aten.slice.Tensor","aten.select.int",
    "aten.unsqueeze.default","aten.squeeze.dim","aten.squeeze.default","aten.transpose.int","aten.swapaxes.default",
    "aten.permute.default","aten.t.default","aten.expand.default","aten.expand_as.default","aten.unbind.int",
    "aten.chunk.default","aten.split.Tensor","aten.split_with_sizes.default","aten.detach.default","aten.alias.default",
    "aten.lift_fresh.default","aten.as_strided.default","aten.narrow.default","aten.narrow.Tensor",
    "aten.reshape.default","aten.flatten.using_ints","aten.unflatten.int"];
/// The eager elementwise call a node makes (code, tensor operands a/b/c, scalar, alpha, scalar_b),
/// mirroring `execute`'s dispatch, when the node is one `tensor::fused` can run.
fn fusible_op(op:&str,node:&Value)->Option<(i32,[Option<String>;3],f32,f32,bool)>{
    let args=node["args"].as_array()?;
    if node["kwargs"].get("out").is_some_and(|v|v.get("tensor").is_some()){return None;}
    let tensor=|v:Option<&Value>|v.and_then(|v|v.get("tensor")).and_then(Value::as_str).map(String::from);
    let a=tensor(args.first())?;
    match op{
        "aten.add.Tensor"|"aten.add.Scalar"|"aten.mul.Tensor"|"aten.mul.Scalar"|"aten.sub.Tensor"|"aten.sub.Scalar"|
        "aten.div.Tensor"|"aten.div.Scalar"|"aten.rsub.Scalar"=>{
            let code=if op.starts_with("aten.add"){20}else if op.starts_with("aten.mul"){21}
                else if op.contains("rsub."){24}else if op.contains("sub."){22}else{23};
            let b=tensor(args.get(1));
            if b.is_none() && args.get(1).and_then(Value::as_f64).is_none(){return None;}
            let scalar=args[1].as_f64().unwrap_or(0.) as f32;
            let alpha=node["kwargs"]["alpha"].as_f64().or_else(||args.get(2).and_then(Value::as_f64)).unwrap_or(1.) as f32;
            let scalar_b=b.is_none();
            Some((code,[Some(a),b,None],scalar,alpha,scalar_b))
        }
        "aten.neg.default"|"aten.sigmoid.default"|"aten.rsqrt.default"|"aten.silu.default"|"aten.gelu.default"|
        "aten.pow.Tensor_Scalar"|"aten.sqrt.default"|"aten.exp.default"|"aten.cos.default"|"aten.sin.default"|
        "aten.log.default"|"aten.reciprocal.default"=>{
            let code=match op{"aten.neg.default"=>0,"aten.sigmoid.default"=>1,"aten.rsqrt.default"=>2,
                "aten.silu.default"=>3,"aten.gelu.default"=>4,"aten.pow.Tensor_Scalar"=>10,
                "aten.sqrt.default"=>6,"aten.exp.default"=>7,"aten.cos.default"=>8,
                "aten.log.default"=>11,"aten.reciprocal.default"=>12,_=>9};
            if op=="aten.gelu.default" && node["kwargs"]["approximate"].as_str()
                .or_else(||args.get(1).and_then(Value::as_str))!=Some("tanh"){return None;}
            let exponent=args.get(1).and_then(Value::as_f64).unwrap_or(0.) as f32;
            let code=if code==10{match exponent{2.=>5,-0.5=>2,0.5=>6,-1.=>12,_=>code}}else{code};
            Some((code,[Some(a),None,None],exponent,1.,true))
        }
        "aten.addcmul.default"=>{
            let alpha=node["kwargs"]["value"].as_f64().unwrap_or(1.) as f32;
            Some((25,[Some(a),Some(tensor(args.get(1))?),Some(tensor(args.get(2))?)],0.,alpha,false))
        }
        "aten.clone.default"|"aten._to_copy.default"|"aten.contiguous.default"|"aten.to.device"|
        "aten.to.dtype"|"aten.to.dtype_layout"|"aten.type_as.default"=>Some((50,[Some(a),None,None],0.,1.,true)),
        _=>None,
    }
}
/// Groups fusible nodes into runs, one open run per output shape: a node joins the open run of its
/// shape (bf16/fp32 operands, at most 8 inputs, 24 ops and 4 outputs). A run closes when another
/// node needs one of its values (or at any node that is neither fusible nor an alias); alias nodes
/// of other values may sit between its nodes. Returns each node's run and the runs.
fn plan_fusion(nodes:&[Value],externals:&serde_json::Map<String,Value>,last:&BTreeMap<String,usize>)
    ->Result<(Vec<Option<usize>>,Vec<FusedRun>)>{
    #[derive(Clone,Copy)] enum Reg{In(usize),Op(usize)}
    struct Building{shape:Vec<usize>,members:Vec<usize>,inputs:Vec<String>,internal:BTreeMap<String,usize>,
        ops:Vec<(i32,[Option<Reg>;3],i32,f32,f32,bool)>,specs:Vec<Value>}
    fn uses(value:&Value,found:&mut BTreeSet<String>){
        match value{
            Value::Object(fields)=>{if let Some(name)=fields.get("tensor").and_then(Value::as_str){found.insert(name.into());}
                else{for field in fields.values(){uses(field,found);}}}
            Value::Array(values)=>for value in values{uses(value,found);},_=>{}
        }
    }
    let mut meta:BTreeMap<String,(Vec<usize>,String)>=BTreeMap::new();
    for (key,spec) in externals{meta.insert(key.clone(),(usize_list(&spec["shape"])?,spec["dtype"].as_str().unwrap_or("").into()));}
    let mut member=vec![None;nodes.len()];let mut runs=Vec::new();
    let close=|b:Building,member:&mut Vec<Option<usize>>,runs:&mut Vec<FusedRun>|{
        if b.members.len()<2{return;}
        let end=*b.members.last().unwrap();
        let resolve=|r:Reg|->i32{match r{Reg::In(i)=>i as i32,Reg::Op(k)=>(b.inputs.len()+k) as i32}};
        let mut outputs=Vec::new();
        for (k,spec) in b.specs.iter().enumerate(){
            let name=spec["tensor"].as_str().unwrap_or("");
            if last.get(name).is_some_and(|&l|l>end){outputs.push((spec.clone(),(b.inputs.len()+k) as i32));}
        }
        if outputs.is_empty() || outputs.len()>4{return;}
        let ops=b.ops.iter().map(|&(op,regs,round,scalar,alpha,scalar_b)|tensor::FuseOp{op,
            a:regs[0].map_or(-1,resolve),b:regs[1].map_or(-1,resolve),c:regs[2].map_or(-1,resolve),
            round,scalar,alpha,scalar_b:scalar_b.into()}).collect();
        for &i in &b.members{member[i]=Some(runs.len());}
        runs.push(FusedRun{end,inputs:b.inputs,outputs,ops});
    };
    let mut open:Vec<Building>=Vec::new();
    for (index,node) in nodes.iter().enumerate(){
        let op=node["op"].as_str().ok_or("operator name")?;
        let mut outs=Vec::new();specs(&node["out"],&mut outs);
        let mut consumed=BTreeSet::new();
        uses(&node["args"],&mut consumed);uses(&node["kwargs"],&mut consumed);
        // Ops with an fp32 result stay eager (casts excepted): in straight-line code the compiler may
        // contract one op's final multiply into the next op's add, which separate kernels (and a
        // bf16 result's rounding instruction) prevent.
        let candidate=(if outs.len()==1{fusible_op(op,node)}else{None}).filter(|(code,operands,..)|{
            let dtype_ok=|d:&str|matches!(d,"torch.bfloat16"|"torch.float32");
            let out=outs[0]["dtype"].as_str().unwrap_or("");
            dtype_ok(out) && (out=="torch.bfloat16" || *code==50)
                && operands.iter().flatten().all(|n|meta.get(n).is_some_and(|(_,d)|dtype_ok(d)))
        });
        if let Some((code,operands,scalar,alpha,scalar_b))=candidate{
            let shape=usize_list(&outs[0]["shape"])?;
            // Values of other-shaped runs must be stored before this node reads them.
            let mut k=0;
            while k<open.len(){
                if open[k].shape!=shape && consumed.iter().any(|n|open[k].internal.contains_key(n)){
                    let b=open.remove(k);close(b,&mut member,&mut runs);
                }else{k+=1;}
            }
            if let Some(k)=open.iter().position(|b|b.shape==shape){
                let b=&open[k];
                let fresh=operands.iter().flatten().filter(|n|!b.internal.contains_key(*n) && !b.inputs.contains(*n))
                    .collect::<BTreeSet<_>>().len();
                if b.inputs.len()+fresh>8 || b.ops.len()>=24{let b=open.remove(k);close(b,&mut member,&mut runs);}
            }
            let k=match open.iter().position(|b|b.shape==shape){Some(k)=>k,None=>{
                open.push(Building{shape:shape.clone(),members:Vec::new(),inputs:Vec::new(),
                    internal:BTreeMap::new(),ops:Vec::new(),specs:Vec::new()});open.len()-1}};
            let b=&mut open[k];
            let mut regs=[None;3];
            for (slot,operand) in operands.iter().enumerate(){
                let Some(name)=operand else{continue};
                regs[slot]=Some(if let Some(&k)=b.internal.get(name){Reg::Op(k)}
                    else if let Some(i)=b.inputs.iter().position(|n|n==name){Reg::In(i)}
                    else{b.inputs.push(name.clone());Reg::In(b.inputs.len()-1)});
            }
            let round=if outs[0]["dtype"]=="torch.bfloat16"{0}else{1};
            b.internal.insert(name(outs[0])?.into(),b.ops.len());
            b.ops.push((code,regs,round,scalar,alpha,scalar_b));b.specs.push(outs[0].clone());b.members.push(index);
        }else if ALIAS_OPS.contains(&op){
            let mut k=0;
            while k<open.len(){
                if consumed.iter().any(|n|open[k].internal.contains_key(n)){let b=open.remove(k);close(b,&mut member,&mut runs);}
                else{k+=1;}
            }
        }else{for b in open.drain(..){close(b,&mut member,&mut runs);}}
        for spec in outs{meta.insert(name(spec)?.into(),(usize_list(&spec["shape"])?,spec["dtype"].as_str().unwrap_or("").into()));}
    }
    for b in open.drain(..){close(b,&mut member,&mut runs);}
    Ok((member,runs))
}

/// A component program, parsed once per process (profiles are immutable), and the names of the
/// externals it may write: arguments of in-place, `out=`, `copy_` and Triton nodes, through views.
fn component_program(path:&str)->Result<(std::rc::Rc<Value>,std::rc::Rc<BTreeSet<String>>)>{
    thread_local!{static PROGRAMS:std::cell::RefCell<BTreeMap<String,(std::rc::Rc<Value>,std::rc::Rc<BTreeSet<String>>)>>=
        const{std::cell::RefCell::new(BTreeMap::new())};}
    if let Some(found)=PROGRAMS.with(|p|p.borrow().get(path).cloned()){return Ok(found);}
    let program:Value=serde_json::from_slice(&std::fs::read(path)?)?;
    // source[v]: the externals a value views.
    let mut source:BTreeMap<String,BTreeSet<String>>=BTreeMap::new();
    for name in program["externals"].as_object().ok_or("component externals")?.keys(){
        source.insert(name.clone(),BTreeSet::from([name.clone()]));
    }
    let mut mutated=BTreeSet::new();
    for node in program["nodes"].as_array().ok_or("component nodes")?{
        let op=node["op"].as_str().ok_or("operator name")?;
        let args=node["args"].as_array().ok_or("operator args")?;
        let tensor_of=|v:&Value|v.get("tensor").and_then(Value::as_str).map(String::from);
        let mut outs=Vec::new();specs(&node["out"],&mut outs);
        let inplace=op.split('.').nth(1).is_some_and(|s|s.ends_with('_'));
        let mut written=Vec::new();
        if inplace || op=="aten.copy_.default"{written.extend(args.first().and_then(tensor_of));}
        if let Some(target)=node["kwargs"].get("out").and_then(tensor_of){written.push(target);}
        if op=="ltx25.triton"{written.extend(args.iter().filter_map(tensor_of));}
        for name in written{if let Some(externals)=source.get(&name){mutated.extend(externals.iter().cloned());}}
        if ALIAS_OPS.contains(&op) || inplace || op=="ltx25.triton"{
            let viewed:BTreeSet<String>=args.iter().filter_map(tensor_of).filter_map(|n|source.get(&n).cloned()).flatten().collect();
            if !viewed.is_empty(){for spec in outs{source.insert(name(spec)?.into(),viewed.clone());}}
        }
    }
    let value=(std::rc::Rc::new(program),std::rc::Rc::new(mutated));
    PROGRAMS.with(|p|p.borrow_mut().insert(path.into(),value.clone()));
    Ok(value)
}

/// Gathers attention operands across the GPUs of a sequence-parallel forward (`sp::Exchange`).
pub(crate) trait SequenceGather {
    fn gather(&self,locals:&[&Tensor],device:&Device)->Result<Vec<Tensor>>;
}

pub fn execute(graph:&Value,tensors:BTreeMap<String,Tensor>,device:&Device,linear:&Linear,
    verify_nodes:bool)->Result<(BTreeMap<String,Tensor>,Value)>{
    execute_with(graph,tensors,device,linear,verify_nodes,None)
}

/// How `execute` runs a program: which nodes run fused, and after which node each value is released.
/// It depends only on the program, so a caller that runs one program many times (the 48 blocks of
/// a forward) plans it once.
pub(crate) struct Plan {member:Vec<Option<usize>>,runs:Vec<FusedRun>,release:Vec<Vec<String>>,verify_nodes:bool}
pub(crate) fn plan(graph:&Value,verify_nodes:bool)->Result<Plan>{
    let setup=host_profiling().then(std::time::Instant::now);
    // Video VAE graphs contain full-resolution temporaries. Release an SSA
    // value after its last consumer; aliased storage stays alive through Rc.
    fn uses(value:&Value,index:usize,last:&mut BTreeMap<String,usize>){
        match value{
            Value::Object(fields)=>{if let Some(name)=fields.get("tensor").and_then(Value::as_str){
                last.insert(name.into(),index);
            }else{for field in fields.values(){uses(field,index,last);}}}
            Value::Array(values)=>for value in values{uses(value,index,last);},_=>{}
        }
    }
    let nodes=graph["nodes"].as_array().ok_or("graph nodes")?;
    let mut last=BTreeMap::new();
    for (index,node) in nodes.iter().enumerate(){
        uses(&node["args"],index,&mut last);uses(&node["kwargs"],index,&mut last);
        let mut outputs=Vec::new();specs(&node["out"],&mut outputs);
        for spec in outputs{last.entry(name(spec)?.into()).or_insert(index);}
    }
    for output in graph["outputs"].as_array().ok_or("graph outputs")?{
        last.insert(output.as_str().ok_or("output id")?.into(),usize::MAX);
    }
    let (member,runs)=if fusion_enabled() && !verify_nodes{
        plan_fusion(nodes,graph["externals"].as_object().ok_or("graph externals")?,&last)?
    }else{(vec![None;nodes.len()],Vec::new())};
    // A run reads its inputs when its last node executes: keep them until then.
    for run in &runs{for input in &run.inputs{if let Some(l)=last.get_mut(input){*l=(*l).max(run.end);}}}
    let mut release=vec![Vec::new();nodes.len()];
    for (name,index) in last{if index<nodes.len(){release[index].push(name);}}
    if let Some(t)=setup{host_record("graph.setup",t);}
    Ok(Plan{member,runs,release,verify_nodes})
}

/// `execute`, with the other GPU's tokens for attentions marked `sp_gather` (a local program of
/// `sp::localize`).
pub(crate) fn execute_with(graph:&Value,tensors:BTreeMap<String,Tensor>,device:&Device,linear:&Linear,
    verify_nodes:bool,gather:Option<&dyn SequenceGather>)->Result<(BTreeMap<String,Tensor>,Value)>{
    execute_planned(graph,&plan(graph,verify_nodes)?,tensors,device,linear,gather)
}

/// `execute_with` of a program planned by `plan` (for this same program).
pub(crate) fn execute_planned(graph:&Value,plan:&Plan,mut tensors:BTreeMap<String,Tensor>,device:&Device,linear:&Linear,
    gather:Option<&dyn SequenceGather>)->Result<(BTreeMap<String,Tensor>,Value)>{
    let verify_nodes=plan.verify_nodes;
    let (member,runs,release)=(&plan.member,&plan.runs,&plan.release);
    let nodes=graph["nodes"].as_array().ok_or("graph nodes")?;
    if member.len()!=nodes.len(){return Err("execution plan belongs to another program".into());}
    let mut comparisons=Vec::new();
    let mut convolution=None;
    let mut triton_kernels=BTreeMap::new();
    for (index,node) in nodes.iter().enumerate(){
        let timer=host_profiling().then(std::time::Instant::now);
        let op=node["op"].as_str().ok_or("operator name")?;
        if let Some(run)=member[index].map(|r|&runs[r]){
            if index==run.end{
                let outputs=run.outputs.iter().map(|(spec,_)|alloc(spec,device)).collect::<Result<Vec<_>>>()?;
                {
                    let inputs=run.inputs.iter().map(|n|tensors.get(n).ok_or_else(||format!("undefined fused input {n}")))
                        .collect::<std::result::Result<Vec<_>,_>>()?;
                    let regs=run.outputs.iter().map(|(_,r)|*r).collect::<Vec<_>>();
                    tensor::fused(&inputs,&outputs.iter().collect::<Vec<_>>(),&regs,&run.ops,device.stream)?;
                }
                for ((spec,_),value) in run.outputs.iter().zip(outputs){tensors.insert(name(spec)?.into(),value);}
            }
            for name in &release[index]{tensors.remove(name);}
            if let Some(t)=timer{host_record(if index==run.end{"fused"}else{"fused.member"},t);}
            continue;
        }
        let args=node["args"].as_array().ok_or("operator args")?;
        let mut outputs=Vec::new();specs(&node["out"],&mut outputs);
        if outputs.is_empty(){return Err(format!("operator {op} has no implemented tensor result").into());}
        let alias_op=matches!(op,"aten.view.default"|"aten._unsafe_view.default"|"aten.slice.Tensor"|
            "aten.select.int"|"aten.unsqueeze.default"|"aten.squeeze.dim"|"aten.squeeze.default"|
            "aten.transpose.int"|"aten.swapaxes.default"|"aten.permute.default"|"aten.t.default"|"aten.expand.default"|"aten.expand_as.default"|
            "aten.unbind.int"|"aten.chunk.default"|"aten.split.Tensor"|"aten.split_with_sizes.default"|"aten.detach.default"|"aten.detach_.default"|
            "aten.alias.default"|"aten.lift_fresh.default"|"aten.as_strided.default"|"aten.dropout.default"|"aten.narrow.default"|"aten.narrow.Tensor");
        let values:Vec<Tensor>=if op=="ltx25.randn"{
            if outputs.len()!=1{return Err("random output count".into());}
            let value=alloc(outputs[0],device)?;
            if !value.is_contiguous() || !matches!(value.dtype.as_str(),"torch.bfloat16"|"torch.float32"){
                return Err("random tensor layout/dtype outside native port".into());
            }
            let metadata=&node["kwargs"];
            let mut generator=crate::rng::CudaGenerator::new(metadata["seed"].as_u64().ok_or("random seed")?)?;
            generator.offset=metadata["offset_before"].as_u64().ok_or("random offset")?;
            generator.randn(value.ptr(),value.elements()?,value.dtype=="torch.bfloat16",device.stream)?;
            if metadata["offset_after"].as_u64()!=Some(generator.offset){return Err("random offset advance differs".into());}
            vec![value]
        }else if op=="ltx25.component"{
            let metadata=&node["kwargs"];
            let (program,mutated)=component_program(metadata["program"].as_str().ok_or("component program")?)?;
            let program=&*program;
            let names=metadata["names"].as_array().ok_or("component bindings")?;
            if names.len()!=args.len(){return Err("component binding count differs".into());}
            let mut inputs=BTreeMap::new();
            for (name,arg) in names.iter().zip(args){
                let name=name.as_str().ok_or("component external name")?;
                let value=get(&tensors,arg)?;
                let spec=&program["externals"][name];let restored=alloc(spec,device)?;
                if restored.shape!=value.shape || restored.dtype!=value.dtype{return Err("component input differs".into());}
                tensor::copy(value,&restored,device.stream)?;inputs.insert(name.to_owned(),restored);
            }
            for (name,constant) in metadata["constants"].as_object().ok_or("component constants")?{
                let entry=&constant["entry"];let path=entry["file"].as_str().ok_or("constant file")?;
                let sha256=entry["sha256"].as_str().ok_or("constant SHA")?;
                // A constant the program never writes (directly or through a view) is uploaded once
                // per process; one it may write gets a fresh copy every call.
                let cache_key=(!mutated.contains(name)).then(||format!("component-constant\0{sha256}\0{}",constant["layout"]));
                if let Some(value)=cache_key.as_ref().and_then(|key|device.cached_weight(key)){inputs.insert(name.clone(),value);continue;}
                let raw=device.asset(Path::new(path),sha256,
                    entry["bytes"].as_u64().and_then(|n|usize::try_from(n).ok()).ok_or("constant bytes")?)?;
                let flat=Tensor::contiguous_shape(usize_list(&entry["shape"])?,entry["dtype"].as_str().ok_or("constant dtype")?.into(),device)?;
                crate::cuda::upload_into(&flat.storage,&raw,device)?;
                let value=alloc(&constant["layout"],device)?;
                tensor::copy(&flat,&value,device.stream)?;
                if let Some(key)=cache_key{device.cache_weight(key,&value);}
                inputs.insert(name.clone(),value);
            }
            let missing=external_names(&program)?.into_iter().filter(|name|!inputs.contains_key(name)).collect::<Vec<_>>();
            if !missing.is_empty(){
                return Err(format!("unbound component external in {}: {missing:?}; gold activation fallback is forbidden",metadata["program"]).into());
            }
            let (values,report)=execute(&program,inputs,device,linear,false)?;
            if report["executed"]!=true{return Err("native component did not finish".into());}
            eprintln!("LTX25_NATIVE_COMPONENT node={index} program={} passed=true",metadata["program"]);
            let ids=program["outputs"].as_array().ok_or("component outputs")?;
            if ids.len()!=outputs.len(){return Err("component result count differs".into());}
            ids.iter().zip(&outputs).map(|(id,spec)|{
                let mut value=values.get(id.as_str().ok_or("component result id")?).ok_or("component native output")?.clone();
                if json!(value.shape)!=spec["shape"] || json!(value.strides)!=spec["stride"]{
                    return Err("component output layout differs".into());
                }
                value.reference_offset=spec["storage_offset"].as_u64().unwrap_or(0) as usize;Ok(value)
            }).collect::<Result<_>>()?
        }else if op=="ltx25.triton"{
            let key=node["kwargs"]["cubin_sha256"].as_str().ok_or("Triton CUBIN SHA")?;
            if !triton_kernels.contains_key(key){
                let root=node["kwargs"]["kernel_root"].as_str().ok_or("Triton kernel root")?;
                triton_kernels.insert(key.to_string(),device.kernel(Path::new(root),key)?);
            }
            let kernel=&triton_kernels[key];
            if kernel.params.len()!=args.len(){return Err("Triton argument count differs".into());}
            let mut params=Vec::new();let mut values=Vec::new();let mut output=0;
            for (arg,kind) in args.iter().zip(&kernel.params){
                use crate::triton::Arg;
                if kind.starts_with('*'){
                    let input=get(&tensors,arg)?;
                    params.push(Arg::Ptr(input.ptr()));
                    values.push(alias(input,outputs.get(output).ok_or("Triton pointer output")?)?);output+=1;
                }else{
                    params.push(match kind.as_str(){
                        "i32"=>Arg::I32(arg.as_i64().ok_or("Triton i32")?.try_into()?),
                        "i64"=>Arg::I64(arg.as_i64().ok_or("Triton i64")?),
                        "fp32"=>Arg::F32(arg.as_f64().ok_or("Triton fp32")? as f32),
                        _=>return Err(format!("unported Triton ABI type {kind}").into())});
                }
            }
            if values.len()!=outputs.len(){return Err("Triton output aliases differ".into());}
            let grid=usize_list(&node["kwargs"]["grid"])?;
            if grid.len()!=3{return Err("Triton grid rank".into());}
            kernel.launch([grid[0].try_into()?,grid[1].try_into()?,grid[2].try_into()?],&params,device.stream)?;
            values
        }else if matches!(op,"aten.reshape.default"|"aten.flatten.using_ints"|"aten.unflatten.int") && outputs.len()==1 && {
            let source=get(&tensors,&args[0])?;let spec=outputs[0];
            spec["storage_offset"].as_u64()==Some(source.reference_offset as u64) &&
                storage_runs(&source.shape,&source.strides)==storage_runs(&usize_list(&spec["shape"])?,&usize_list(&spec["stride"])?)
        }{
            vec![alias(get(&tensors,&args[0])?,outputs[0])?]
        }else if alias_op {
            if op=="aten.dropout.default" && args.get(2).and_then(Value::as_bool).unwrap_or(true)
                && args.get(1).and_then(Value::as_f64).unwrap_or(1.)!=0.{return Err("training dropout is not ported".into());}
            let source=get(&tensors,&args[0])?;
            outputs.iter().map(|s|alias(source,s)).collect::<Result<_>>()?
        } else {
            let inplace=op.split('.').nth(1).is_some_and(|s|s.ends_with('_'));
            let target=node["kwargs"].get("out").filter(|v|v.get("tensor").is_some())
                .or_else(||if inplace{args.first()}else{None});
            let values=outputs.iter().map(|s|if let Some(target)=target{alias(get(&tensors,target)?,s)}else{alloc(s,&device)})
                .collect::<Result<Vec<_>>>()?;
            let y=&values[0];
            match op {
                "ltx25.natten3d"=>{
                    if node["kwargs"]["backend"]!="cutlass-fna"{return Err("unported NATTEN backend".into());}
                    crate::natten::forward(get(&tensors,&args[0])?,get(&tensors,&args[1])?,get(&tensors,&args[2])?,y,
                        &usize_list(&node["kwargs"]["kernel_size"])?,&usize_list(&node["kwargs"]["q_tile"])?,
                        &usize_list(&node["kwargs"]["kv_tile"])?,node["kwargs"]["scale"].as_f64().ok_or("NATTEN scale")? as f32,device)?;
                }
                "aten.empty.memory_format"|"aten.empty_strided.default"|"aten.empty_like.default"|"aten.new_empty.default"=>{}
                "aten.copy_.default"=>{tensor::copy(get(&tensors,&args[1])?,y,device.stream)?;}
                "aten.conv1d.default"|"aten.conv2d.default"|"aten.conv3d.default"|"aten.conv_transpose1d.default"=>{
                    let x=get(&tensors,&args[0])?;let w=get(&tensors,&args[1])?;
                    let bias=args.get(2).filter(|v|!v.is_null()).map(|v|get(&tensors,v)).transpose()?;
                    let spatial=x.shape.len()-2;
                    let stride=args.get(3).map(usize_list).transpose()?.unwrap_or(vec![1;spatial]);
                    let padding=args.get(4).map(usize_list).transpose()?.unwrap_or(vec![0;spatial]);
                    let transposed=op=="aten.conv_transpose1d.default";
                    let dilation=args.get(if transposed{7}else{5}).map(usize_list).transpose()?.unwrap_or(vec![1;spatial]);
                    let groups=args.get(6).and_then(Value::as_u64).unwrap_or(1) as usize;
                    if convolution.is_none(){convolution=Some(crate::convolution::Conv::new()?);}
                    convolution.as_ref().unwrap().forward(x,w,bias,y,&stride,&padding,&dilation,groups,transposed,device)?;
                }
                "aten.pad.default"=>{
                    let pads=args[1].as_array().ok_or("padding array")?.iter().map(|v|v.as_i64().ok_or("padding scalar"))
                        .collect::<std::result::Result<Vec<_>,_>>()?;
                    let mode=args.get(2).and_then(Value::as_str).unwrap_or("constant");
                    let mode=match mode{"constant"=>0,"replicate"=>1,_=>return Err("padding mode not ported".into())};
                    let scalar=args.get(3).and_then(Value::as_f64).unwrap_or(0.) as f32;
                    tensor::spatial(get(&tensors,&args[0])?,y,&pads,mode,scalar,device)?;
                }
                "aten.upsample_nearest2d.vec"=>{
                    let x=get(&tensors,&args[0])?;
                    if x.shape.len()!=4 || x.shape[..2]!=y.shape[..2] || y.shape[2]!=x.shape[2]*2 || y.shape[3]!=x.shape[3]*2{
                        return Err("only exact 2x nearest upsampling is ported".into());
                    }
                    tensor::spatial(x,y,&[],2,0.,device)?;
                }
                "aten.group_norm.default"=>{
                    let x=get(&tensors,&args[0])?;let groups=args[1].as_u64().ok_or("GroupNorm groups")? as usize;
                    let w=args.get(2).filter(|v|!v.is_null()).map(|v|get(&tensors,v)).transpose()?;
                    let bias=args.get(3).filter(|v|!v.is_null()).map(|v|get(&tensors,v)).transpose()?;
                    let eps=args.get(4).and_then(Value::as_f64).unwrap_or(1e-5) as f32;
                    tensor::group_norm(x,w,bias,y,groups,eps,device)?;
                }
                "aten.arange.start"|"aten.arange.start_step"|"aten.arange.default"=>{
                    let start=if op=="aten.arange.default"{0.}else{args[0].as_f64().ok_or("arange start")? as f32};
                    let step=if op=="aten.arange.start_step"{args[2].as_f64().ok_or("arange step")? as f32}else{1.};
                    if start.fract()!=0. || step.fract()!=0.{return Err("fractional arange is not ported".into());}
                    tensor::fill(y,start,step,device.stream)?;
                }
                "aten.ones_like.default"|"aten.zeros_like.default"|"aten.full_like.default"|"aten.new_zeros.default"|
                "aten.zeros.default"|"aten.ones.default"|"aten.full.default"=>{
                    let value=match op{"aten.ones_like.default"|"aten.ones.default"=>1.,"aten.zeros_like.default"|"aten.zeros.default"=>0.,
                        "aten.new_zeros.default"=>0.,
                        _=>args[1].as_f64().ok_or("full_like scalar")? as f32};
                    tensor::fill(y,value,0.,device.stream)?;
                }
                "aten.clone.default"|"aten._to_copy.default"|"aten.contiguous.default"|"aten.to.device"|
                "aten.to.dtype"|"aten.to.dtype_layout"|"aten.type_as.default"|
                "aten.reshape.default"|"aten.flatten.using_ints"|"aten.unflatten.int"=>{
                    tensor::copy(get(&tensors,&args[0])?,y,device.stream)?;
                }
                "aten.linear.default"=>{
                    let x=get(&tensors,&args[0])?;let w=get(&tensors,&args[1])?;
                    let bias=args.get(2).filter(|v|!v.is_null()).map(|v|get(&tensors,v)).transpose()?;
                    let k=*x.shape.last().ok_or("linear input rank")?;
                    if w.shape.len()!=2 || w.shape[1]!=k || w.dtype!=x.dtype
                        || w.strides!=tensor::contiguous_strides(&w.shape){
                        return Err("unsupported linear layout".into());
                    }
                    if x.is_contiguous(){
                        linear.forward(x.ptr(),w.ptr(),bias.map(Tensor::ptr),y.ptr(),x.elements()?/k,w.shape[0],k,&x.dtype)?;
                    }else if x.shape.len()==3 && x.shape[0]==1 && x.strides[1]==1 && x.strides[2]==x.shape[1]{
                        linear.column_major(x.ptr(),w.ptr(),y.ptr(),x.shape[1],w.shape[0],k,&x.dtype)?;
                        if let Some(bias)=bias{tensor::elementwise(20,y,Some(bias),None,y,0.,1.,device.stream)?;}
                    }else{
                        // ATen linear's noncontiguous N-D matmul flattens a
                        // contiguous copy, rounds GEMM to the input dtype, then
                        // applies bias as a separate eager operation.
                        let materialized=x.contiguous(device)?;
                        linear.forward(materialized.ptr(),w.ptr(),None,y.ptr(),x.elements()?/k,w.shape[0],k,&x.dtype)?;
                        if let Some(bias)=bias{tensor::elementwise(20,y,Some(bias),None,y,0.,1.,device.stream)?;}
                    }
                }
                "aten.add.Tensor"|"aten.add.Scalar"|"aten.add_.Tensor"|"aten.add_.Scalar"|
                "aten.mul.Tensor"|"aten.mul.Scalar"|"aten.mul_.Tensor"|"aten.mul_.Scalar"|
                "aten.sub.Tensor"|"aten.sub.Scalar"|"aten.div.Tensor"|"aten.div.Scalar"|
                "aten.rsub.Scalar"=>{
                    let code=if op.starts_with("aten.add"){20}else if op.starts_with("aten.mul"){21}
                        else if op.contains("rsub."){24}else if op.contains("sub."){22}else{23};
                    let a=get(&tensors,&args[0])?;
                    let b=if args[1].get("tensor").is_some(){Some(get(&tensors,&args[1])?)}else{None};
                    let scalar=args[1].as_f64().unwrap_or(0.) as f32;
                    let alpha=node["kwargs"]["alpha"].as_f64().or_else(||args.get(2).and_then(Value::as_f64)).unwrap_or(1.) as f32;
                    tensor::elementwise(code,a,b,None,y,scalar,alpha,device.stream)?;
                }
                "aten.neg.default"|"aten.sigmoid.default"|"aten.rsqrt.default"|"aten.silu.default"|"aten.silu_.default"|
                "aten.gelu.default"|"aten.pow.Tensor_Scalar"|"aten.sqrt.default"|"aten.exp.default"|
                "aten.cos.default"|"aten.sin.default"|"aten.log.default"|"aten.reciprocal.default"=>{
                    let code=match op{"aten.neg.default"=>0,"aten.sigmoid.default"=>1,"aten.rsqrt.default"=>2,
                        "aten.silu.default"|"aten.silu_.default"=>3,"aten.gelu.default"=>4,"aten.pow.Tensor_Scalar"=>10,
                        "aten.sqrt.default"=>6,"aten.exp.default"=>7,"aten.cos.default"=>8,
                        "aten.log.default"=>11,"aten.reciprocal.default"=>12,_=>9};
                    if op=="aten.gelu.default" && node["kwargs"]["approximate"].as_str()
                        .or_else(||args.get(1).and_then(Value::as_str))!=Some("tanh"){
                        return Err("only tanh GELU has been ported".into());
                    }
                    let exponent=args.get(1).and_then(Value::as_f64).unwrap_or(0.) as f32;
                    let code=if code==10{match exponent{2.=>5,-0.5=>2,0.5=>6,-1.=>12,_=>code}}else{code};
                    tensor::elementwise(code,get(&tensors,&args[0])?,None,None,
                        y,exponent,1.,device.stream)?;
                }
                "aten.mean.dim"=>{
                    let dims=args[1].as_array().ok_or("mean dimensions")?.iter()
                        .map(|v|v.as_i64().ok_or("mean dimension")).collect::<std::result::Result<Vec<_>,_>>()?;
                    tensor::mean(get(&tensors,&args[0])?,y,&dims,device)?;
                }
                "aten.linalg_vector_norm.default"=>{
                    if args.get(1).and_then(Value::as_f64).unwrap_or(2.)!=2.{return Err("only norm order 2 is ported".into());}
                    let dims=args[2].as_array().ok_or("norm dimensions")?.iter().map(|v|v.as_i64().ok_or("norm dimension"))
                        .collect::<std::result::Result<Vec<_>,_>>()?;
                    tensor::norm2(get(&tensors,&args[0])?,y,&dims,device)?;
                }
                "aten.atan2.default"=>{
                    tensor::elementwise(35,get(&tensors,&args[0])?,Some(get(&tensors,&args[1])?),None,y,0.,1.,device.stream)?;
                }
                "aten.clamp.default"|"aten.clamp_.default"=>{
                    tensor::elementwise(36,get(&tensors,&args[0])?,None,None,y,
                        args.get(1).and_then(Value::as_f64).unwrap_or(f64::NEG_INFINITY) as f32,
                        args.get(2).and_then(Value::as_f64).unwrap_or(f64::INFINITY) as f32,device.stream)?;
                }
                "aten.matmul.default"=>{
                    linear.bmm_nn(get(&tensors,&args[0])?,get(&tensors,&args[1])?,y)?;
                }
                "aten.addcmul.default"=>{
                    let alpha=node["kwargs"]["value"].as_f64().unwrap_or(1.) as f32;
                    tensor::elementwise(25,get(&tensors,&args[0])?,Some(get(&tensors,&args[1])?),
                        Some(get(&tensors,&args[2])?),y,0.,alpha,device.stream)?;
                }
                "aten.gt.Scalar"|"aten.ge.Scalar"|"aten.lt.Scalar"=>{
                    let code=match op{"aten.gt.Scalar"=>30,"aten.ge.Scalar"=>32,_=>33};
                    tensor::elementwise(code,get(&tensors,&args[0])?,None,None,y,
                        args[1].as_f64().ok_or("comparison scalar")? as f32,1.,device.stream)?;
                }
                "aten.where.self"=>{
                    tensor::elementwise(34,get(&tensors,&args[0])?,Some(get(&tensors,&args[1])?),
                        Some(get(&tensors,&args[2])?),y,0.,1.,device.stream)?;
                }
                "aten.gather.default"=>{
                    let mut dim=args[1].as_i64().ok_or("gather axis")?;
                    if dim<0{dim+=y.shape.len() as i64;}
                    tensor::index(0,get(&tensors,&args[0])?,Some(get(&tensors,&args[2])?),y,dim.try_into()?,device)?;
                }
                "aten.repeat.default"=>{
                    let x=get(&tensors,&args[0])?;let repeats=usize_list(&args[1])?;
                    if repeats.len()!=y.shape.len() || repeats.len()<x.shape.len(){return Err("repeat rank differs".into());}
                    for i in 0..repeats.len(){
                        let extent=if i+ x.shape.len()>=repeats.len(){x.shape[i+x.shape.len()-repeats.len()]}else{1};
                        if extent.checked_mul(repeats[i])!=Some(y.shape[i]){return Err("repeat extent differs".into());}
                    }
                    tensor::index(1,x,None,y,0,device)?;
                }
                "aten.argsort.stable"=>{
                    let dim=node["kwargs"]["dim"].as_i64().or_else(||args.get(1).and_then(Value::as_i64)).unwrap_or(-1);
                    if dim!=-1 && dim!=(y.shape.len()-1) as i64 || node["kwargs"]["stable"].as_bool()!=Some(true){
                        return Err("only stable last-axis binary argsort is ported".into());
                    }
                    tensor::binary_sort(get(&tensors,&args[0])?,y,node["kwargs"]["descending"].as_bool().unwrap_or(false),device)?;
                }
                "aten.mm.default"|"aten.mm.out"|"aten.addmm.default"=>{
                    let bias=if op=="aten.addmm.default"{Some(get(&tensors,&args[0])?)}else{None};
                    let i=usize::from(bias.is_some());
                    let x=get(&tensors,&args[i])?;let w=get(&tensors,&args[i+1])?;
                    if x.shape.len()!=2 || w.shape.len()!=2 || x.strides!=tensor::contiguous_strides(&x.shape)
                        || w.strides!=vec![1,w.shape[0]] || x.shape[1]!=w.shape[0] {
                        return Err(format!("unsupported GEMM layout at node {index}").into());
                    }
                    for key in ["alpha","beta"]{if node["kwargs"][key].as_f64().unwrap_or(1.)!=1.{
                        return Err("GEMM alpha/beta other than 1 unported".into());}}
                    linear.forward(x.ptr(),w.ptr(),bias.map(Tensor::ptr),y.ptr(),x.shape[0],w.shape[1],x.shape[1],&x.dtype)?;
                }
                "aten._fused_rms_norm.default"|"aten.rms_norm.default"=>{
                    let x=get(&tensors,&args[0])?;
                    let w=args.get(2).filter(|v|!v.is_null()).map(|v|get(&tensors,v)).transpose()?;
                    let eps=args.get(3).and_then(Value::as_f64).ok_or("RMS epsilon must be explicit")? as f32;
                    tensor::rms(x,w,y,values.get(1),eps,&device)?;
                }
                "aten.layer_norm.default"|"aten.native_layer_norm.default"=>{
                    let x=get(&tensors,&args[0])?;
                    if usize_list(&args[1])?!=vec![*x.shape.last().ok_or("LayerNorm rank")?]{
                        return Err("only last-dimension LayerNorm is ported".into());
                    }
                    let w=args.get(2).filter(|v|!v.is_null()).map(|v|get(&tensors,v)).transpose()?;
                    let bias=args.get(3).filter(|v|!v.is_null()).map(|v|get(&tensors,v)).transpose()?;
                    let eps=args.get(4).and_then(Value::as_f64).ok_or("LayerNorm epsilon")? as f32;
                    tensor::layer_norm(x,w,bias,y,values.get(1),values.get(2),eps,&device)?;
                }
                "aten.cat.default"|"aten.concatenate.default"=>{
                    let items=args[0].as_array().ok_or("cat operands")?;
                    let mut dim=args.get(1).and_then(Value::as_i64).unwrap_or(0);
                    if dim<0{dim+=y.shape.len() as i64;}
                    let dim=usize::try_from(dim)?;let mut offset=0;
                    for item in items{
                        let x=get(&tensors,item)?;if x.elements()?==0{continue;}
                        let mut slice=y.clone();slice.shape=x.shape.clone();
                        slice.offset+=offset*y.strides[dim]*tensor::dtype(&y.dtype)?.1;
                        tensor::copy(x,&slice,device.stream)?;offset+=x.shape[dim];
                    }
                    if offset!=y.shape[dim]{return Err("cat extent differs".into());}
                }
                "aten.stack.default"=>{
                    let items=args[0].as_array().ok_or("stack operands")?;
                    let mut dim=args.get(1).and_then(Value::as_i64).unwrap_or(0);
                    if dim<0{dim+=y.shape.len() as i64;}
                    let dim=usize::try_from(dim)?;
                    if dim>=y.shape.len() || items.len()!=y.shape[dim]{return Err("stack extent differs".into());}
                    for (index,item) in items.iter().enumerate(){
                        let x=get(&tensors,item)?;let mut slice=y.clone();
                        slice.offset+=index*y.strides[dim]*tensor::dtype(&y.dtype)?.1;
                        slice.shape.remove(dim);slice.strides.remove(dim);
                        if slice.shape!=x.shape{return Err("stack input shape differs".into());}
                        tensor::copy(x,&slice,device.stream)?;
                    }
                }
                "aten._scaled_dot_product_cudnn_attention.default"|"aten.scaled_dot_product_attention.default"=>{
                    let q=get(&tensors,&args[0])?;let k=get(&tensors,&args[1])?;let v=get(&tensors,&args[2])?;
                    let gathered=if node["sp_gather"]==true{
                        Some(gather.ok_or("sequence-parallel attention without its other GPU")?.gather(&[k,v],device)?)
                    }else{None};
                    let (k,v)=match &gathered{Some(full)=>(&full[0],&full[1]),None=>(k,v)};
                    let view=|t:&Tensor|->Result<sdpa::View>{Ok(sdpa::View{dims:t.shape.iter().map(|&x|x as i64).collect::<Vec<_>>()
                        .try_into().map_err(|_|"SDPA requires rank four")?,
                        strides:t.strides.iter().map(|&x|x as i64).collect::<Vec<_>>().try_into().map_err(|_|"SDPA rank")?,offset_bytes:0})};
                    let (qv,kv,vv)=(view(q)?,view(k)?,view(v)?);
                    if q.dtype!="torch.bfloat16" || k.dtype!=q.dtype || v.dtype!=q.dtype{
                        return Err("SDPA dtype not ported".into());
                    }
                    let mask=args.get(3).filter(|v|!v.is_null()).or_else(||node["kwargs"].get("attn_mask").filter(|v|!v.is_null()))
                        .map(|v|get(&tensors,v)).transpose()?;
                    let scale=node["kwargs"]["scale"].as_f64().map(|x|x as f32)
                        .unwrap_or_else(||sdpa::default_scale(qv.dims[3]));
                    if node["sdpa_backend"].as_i64()==Some(2){
                        sdpa::efficient(q,k,v,mask,y,scale,device)?;
                    }else{
                    if mask.is_some_and(|m|m.dtype!=q.dtype){return Err("cuDNN SDPA requires additive BF16 bias".into());}
                    // attention.cu::_cudnn_attention_forward expands bias to
                    // [B, original_mask_heads, Sq, Sk] before building the graph.
                    let mut expanded_mask=mask.cloned();
                    if let Some(m)=&mut expanded_mask{
                        if m.shape.len()!=4{return Err("cuDNN bias rank must be four".into());}
                        for (axis,extent) in [(0,q.shape[0]),(2,q.shape[2]),(3,k.shape[2])]{
                            if m.shape[axis]!=extent{
                                if m.shape[axis]!=1{return Err("cuDNN bias is not broadcastable".into());}
                                m.shape[axis]=extent;m.strides[axis]=0;
                            }
                        }
                    }
                    let mask=expanded_mask.as_ref();
                    let mask_view=mask.map(view).transpose()?;
                    let plan=sdpa::prepare(sdpa::Io::Bf16,&qv,&kv,&vv,mask_view.as_ref())?;
                    if plan.o_strides.iter().map(|&n|n as usize).collect::<Vec<_>>()!=y.strides{
                        return Err("SDPA output strides differ from the reference".into());
                    }
                    // A stream-ordered transient: the next user of these bytes runs after this attention
                    // on the same stream, so neither a free nor a device synchronization is needed.
                    let workspace=if plan.workspace>0{Some(device.transient(plan.workspace)?)}else{None};
                    sdpa::forward(sdpa::Io::Bf16,(q.ptr(),&qv),(k.ptr(),&kv),(v.ptr(),&vv),y.ptr(),
                        workspace.as_ref(),mask.zip(mask_view.as_ref()).map(|(m,v)|(m.ptr(),v)),scale,device.stream)?;
                    }
                    if values.iter().skip(1).any(|t|t.elements().unwrap_or(1)>0){
                        return Err("SDPA statistics/random outputs are not ported".into());
                    }
                }
                _=>return Err(format!("unported operator {op} at node {index}").into()),
            }
            values
        };
        for (value,spec) in values.into_iter().zip(outputs){
            if verify_nodes && spec["uninitialized"]!=true {
                let actual=value.download(&device)?;
                let passed=sha(&actual)==spec["sha256"].as_str().ok_or("missing node SHA256")?;
                comparisons.push(json!({"node":index,"op":op,"tensor":name(spec)?,"passed":passed,
                    "sha256":sha(&actual),"expected_sha256":spec["sha256"]}));
                if !passed{return Ok((tensors,json!({"passed":false,"checks":comparisons,"first_mismatch":index})));}
            }
            tensors.insert(name(spec)?.into(),value);
        }
        for name in &release[index]{tensors.remove(name);}
        if let Some(t)=timer{host_record(op,t);}
    }
    // Results are stream-ordered for the caller's next work (downloads synchronize themselves), so a
    // finished graph does not drain the GPU; a per-node verification run does, for its report.
    if verify_nodes{
        let synced=host_profiling().then(std::time::Instant::now);
        device.sync()?;
        if let Some(t)=synced{host_record("graph.sync",t);}
    }
    Ok((tensors,json!({"passed":if verify_nodes{json!(!comparisons.is_empty())}else{Value::Null},
        "executed":true,"rung":"layer","checks":comparisons})))
}
