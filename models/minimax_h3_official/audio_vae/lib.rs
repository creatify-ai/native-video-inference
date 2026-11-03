//! Explicit no-Python execution of an original audio decoder shape graph.
#[path="../support/startup_trace.rs"]
pub mod startup_trace;
#[path="../support/weight_loader.rs"]
pub mod weight_loader;
mod allocation_pool;
mod audio_delivery;
#[path="../support/light_sha.rs"] mod light_sha;
fn weight_digest_matches(bytes:&[u8],expected:&str)->Result<bool>{
    if light_sha::async_enabled()? {light_sha::verify_model_bytes(bytes,expected)?;Ok(true)}
    else {Ok(digest(bytes)==expected)}
}
use serde::{Deserialize,Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, ffi::{c_char,c_void,CStr}, fs, path::{Path,PathBuf}, ptr, rc::Rc};

pub type Result<T> = std::result::Result<T,String>;
pub type DevicePtr = *mut c_void;
extern "C" {
    fn cudaMalloc(pointer: *mut DevicePtr, count: usize) -> i32;
    fn cudaFree(pointer: DevicePtr) -> i32;
    fn cudaMemcpy(dst: DevicePtr, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(dst: DevicePtr, src: *const c_void, count: usize, kind: i32, stream:DevicePtr) -> i32;
    fn cudaStreamSynchronize(stream: DevicePtr) -> i32;
    fn cudaGetErrorString(code: i32) -> *const c_char;
    fn h3_conv_create(context: *mut DevicePtr) -> i32;
    fn h3_conv_forward(context:DevicePtr,spec:*const ConvSpec,input:*const c_void,weight:*const c_void,output:DevicePtr,stream:DevicePtr)->i32;
    fn h3_conv_error()->*const c_char;
    fn h3_conv_plan_tag(context:DevicePtr)->*const c_char;
    fn h3_conv_destroy(context:DevicePtr)->i32;
    fn h3_conv_cache_stats(context:DevicePtr,out:*mut u64)->i32;
    fn h3_audio_pointwise(op:i32,a:*const c_void,b:*const c_void,out:DevicePtr,count:i64,shape:*const i64,astride:*const i64,bstride:*const i64,sa:f32,sb:f32,stream:DevicePtr)->i32;
    fn h3_audio_pad(input:*const c_void,out:DevicePtr,count:i64,input_length:i32,output_length:i32,left:i32,stream:DevicePtr)->i32;
    fn h3_audio_depthwise(input:*const c_void,weight:*const c_void,bias:*const c_void,out:DevicePtr,count:i32,channels:i32,input_length:i32,output_length:i32,kernel:i32,stride:i32,padding:i32,dilation:i32,stream:DevicePtr)->i32;
}
fn cuda(code:i32)->Result<()> { if code==0 {Ok(())} else {Err(unsafe{CStr::from_ptr(cudaGetErrorString(code)).to_string_lossy().into_owned()})} }
fn conv(code:i32)->Result<()> { if code==0 {Ok(())} else {Err(unsafe{CStr::from_ptr(h3_conv_error()).to_string_lossy().into_owned()})} }
pub fn digest(bytes:&[u8])->String { format!("{:x}",crate::startup_trace::measure("sha256","",(bytes).len(),||Sha256::digest(bytes))) }

struct Allocation { pointer:DevicePtr, owned:bool, _pool_lease:Option<allocation_pool::Lease> }
impl Drop for Allocation { fn drop(&mut self) { if self.owned {unsafe{cudaFree(self.pointer);}} } }
#[derive(Clone)]
pub struct Tensor { memory:Rc<Allocation>, pub shape:Vec<i64> }
impl Tensor {
    fn empty(shape:&[i64])->Result<Self> {
        let mut pointer = ptr::null_mut();
        let count=shape.iter().product::<i64>() as usize*4;
        unsafe{cuda(cudaMalloc(&mut pointer,count))?;}
        Ok(Self{memory:Rc::new(Allocation{pointer,owned:true,_pool_lease:None}),shape:shape.to_vec()})
    }
    fn pooled(shape:&[i64],pool:&allocation_pool::Pool,stream:DevicePtr)->Result<Self> {
        let bytes=shape.iter().product::<i64>() as usize*4;
        let lease=pool.acquire(bytes,stream)?;
        let pointer=lease.pointer();
        Ok(Self{memory:Rc::new(Allocation{pointer,owned:false,_pool_lease:Some(lease)}),shape:shape.to_vec()})
    }
    pub fn from_bytes(shape:&[i64],bytes:&[u8])->Result<Self> {
        let result=Self::empty(shape)?;
        if result.bytes()!=bytes.len(){return Err("tensor byte count mismatch".into());}
        unsafe{cuda(cudaMemcpy(result.pointer(),bytes.as_ptr().cast(),bytes.len(),1))?;}
        Ok(result)
    }
    pub fn pointer(&self)->DevicePtr {self.memory.pointer}
    /// Pack each channel's first `samples` values into a contiguous [2, samples] tensor.
    pub fn prefix_samples(&self,shape:&[i64],samples:usize,stream:DevicePtr)->Result<Self> {
        let ranges=audio_delivery::prefix_ranges(shape,self.bytes(),samples)?;
        if samples as i64==shape[1]{let mut output=self.clone();output.shape=shape.to_vec();return Ok(output);}
        let output=Self::empty(&[2,samples as i64])?;
        for (channel,(first,bytes)) in ranges.into_iter().enumerate() {
            unsafe{cuda(cudaMemcpyAsync(output.pointer().cast::<u8>().add(channel*bytes).cast(),
                self.pointer().cast::<u8>().add(first).cast(),bytes,3,stream))?;}
        }
        Ok(output)
    }
    pub fn bytes(&self)->usize {self.shape.iter().product::<i64>() as usize*4}
    pub fn download(&self,stream:DevicePtr)->Result<Vec<u8>> {
        unsafe{cuda(cudaStreamSynchronize(stream))?;}
        let mut bytes=vec![0;self.bytes()];
        unsafe{cuda(cudaMemcpy(bytes.as_mut_ptr().cast(),self.pointer(),bytes.len(),2))?;}
        Ok(bytes)
    }
}
#[repr(C)]
#[derive(Default)]
struct ConvSpec {
    ndim:i32,dtype:i32,transposed:i32,allow_tf32:i32,deterministic:i32,channels_last:i32,
    input_dims:[i64;5],input_strides:[i64;5],weight_dims:[i64;5],weight_strides:[i64;5],
    output_dims:[i64;5],output_strides:[i64;5],padding:[i64;3],stride:[i64;3],dilation:[i64;3],
}
#[derive(Deserialize,Serialize,Clone)]
struct Spec { file:String,dtype:String,shape:Vec<i64>,bytes:usize,sha256:String }
#[derive(Deserialize,Serialize,Clone)]
struct Node {id:String,op:String,args:Vec<Value>,kwargs:Value,output:Spec}
#[derive(Deserialize,Serialize,Clone)]
struct Manifest {schema:String,status:String,input:Spec,output_id:String,gold:Spec,constants:HashMap<String,Spec>,nodes:Vec<Node>,provenance:Value}
pub struct Decoder {
    root:PathBuf, manifest:Manifest, constants:HashMap<String,Tensor>, context:DevicePtr,
    uses:HashMap<String,usize>, pub plans:HashMap<String,String>,
    allocation_pool:allocation_pool::Pool, allocation_pool_enabled:bool,
    shared_weights:HashMap<String,Tensor>,shape_graphs:HashMap<i64,ShapeGraph>,
    template:Manifest,template_sha:String,weight_identity:String,graph_clock:u64,graph_evictions:usize,
    graph_prepare_count:usize,graph_switch_count:usize,last_graph_prepare_seconds:f64,
    last_pool_released_bytes:usize,last_graph_cache_hit:bool,
}
impl Drop for Decoder {fn drop(&mut self){
    if let Err(error)=self.allocation_pool.synchronize(){eprintln!("audio pool decoder drain: {error}");}
    unsafe{h3_conv_destroy(self.context);}
}}
fn tensor_id(value:&Value)->Option<&str>{value.get("tensor").and_then(Value::as_str)}
fn strides(shape:&[i64])->Vec<i64>{let mut s=vec![1;shape.len()];for d in (0..shape.len().saturating_sub(1)).rev(){s[d]=s[d+1]*shape[d+1];}s}
fn shape3(shape:&[i64])->Result<[i64;3]>{if shape.len()>3{return Err(format!("unsupported rank {shape:?}"));}let mut s=[1;3];s[3-shape.len()..].copy_from_slice(shape);Ok(s)}
fn broadcast(tensor:&Tensor,output:&[i64])->Result<[i64;3]>{
    let source=shape3(&tensor.shape)?;let target=shape3(output)?;let mut s=[source[1]*source[2],source[2],1];
    for d in 0..3 {if source[d]==1{s[d]=0;}else if source[d]!=target[d]{return Err(format!("broadcast {:?} -> {:?}",tensor.shape,output));}}Ok(s)
}
fn dims4(shape:&[i64])->[i64;5]{[shape[0],shape[1],1,shape[2],0]}
fn stride4(shape:&[i64])->[i64;5]{[shape[1]*shape[2],shape[2],shape[2],1,0]}
fn integer(value:&Value)->Result<i64>{value.as_i64().ok_or_else(||format!("expected integer: {value}"))}
fn first(value:&Value)->Result<i64>{integer(value.get(0).ok_or("missing array element")?)}
fn list(value:&Value)->Result<Vec<i64>>{value.as_array().ok_or("expected array")?.iter().map(integer).collect()}

include!("shape_rules.rs");
include!("dynamic_graphs.rs");
include!("encoder.rs");

impl Decoder {
    /// Request-level allocation policy. Constants/input/final output never enter the pool.
    pub fn set_allocation_pool_enabled(&mut self,enabled:bool){self.allocation_pool_enabled=enabled;}
    pub fn release_cached_intermediates(&mut self)->Result<usize>{self.allocation_pool.trim()}
    pub fn allocation_pool_report(&self)->Value {
        let s=self.allocation_pool.snapshot();
        json!({"schema":"h3-audio-allocation-pool-v1","enabled":s.enabled,
            "stream":s.stream.map(|v|format!("0x{v:x}")),"exact_size":true,
            "constants_pooled":false,"input_pooled":false,"final_output_pooled":false,
            "request":{"cuda_malloc_calls":s.request.malloc_calls+s.request.unpooled_node_allocations,
                "pool_malloc_calls":s.request.malloc_calls,"reuse_calls":s.request.reuse_calls,
                "return_calls":s.request.return_calls,"unpooled_node_allocations":s.request.unpooled_node_allocations},
            "total":{"pool_malloc_calls":s.total.malloc_calls,"reuse_calls":s.total.reuse_calls,
                "return_calls":s.total.return_calls,"unpooled_node_allocations":s.total.unpooled_node_allocations},
            "cached_buffers":s.cached_buffers,"cached_bytes":s.cached_bytes,
            "live_buffers":s.live_buffers,"live_bytes":s.live_bytes,
            "allocated_bytes":s.allocated_bytes,"high_water_bytes":s.high_water_bytes,
            "counter_scope":"graph node output allocations only; input/constants/library workspaces excluded",
            "high_water_scope":"pool-owned cached plus live device bytes; not GPU kernel peak"})
    }
    pub fn gold_sha(&self)->&str{&self.manifest.gold.sha256}
    pub fn input_sha(&self)->&str{&self.manifest.input.sha256}
    pub fn graph_node_count(&self)->usize{self.manifest.nodes.len()}
    pub fn input_shape(&self)->&[i64]{&self.manifest.input.shape}
    pub fn output_shape(&self)->&[i64]{&self.manifest.gold.shape}
    /// Caller provides normalized FP32 [2,32,T]. All intermediate values are computed natively.
    /// With `diagnostics`, golden checkpoints are read only AFTER computed output exists.
    pub unsafe fn run(&mut self,input:DevicePtr,stream:DevicePtr,diagnostics:Option<&Path>)->Result<Tensor>{
        self.allocation_pool.begin(self.allocation_pool_enabled,stream)?;
        let mut tensors=self.constants.clone();
        tensors.insert("input".into(),Tensor{memory:Rc::new(Allocation{pointer:input,owned:false,_pool_lease:None}),shape:self.manifest.input.shape.clone()});
        let mut uses=self.uses.clone();
        let nodes=self.manifest.nodes.clone();
        for (index,node) in nodes.iter().enumerate(){
            if node.output.dtype!="torch.float32"{return Err(format!("{}: non-FP32 dynamic node",node.id));}
            // Only private intermediates are pooled. The final public Tensor
            // keeps its original independent allocation and cudaFree lifetime.
            let output=if self.allocation_pool.enabled()&&node.id!=self.manifest.output_id {
                Tensor::pooled(&node.output.shape,&self.allocation_pool,stream)?
            }else{
                let output=Tensor::empty(&node.output.shape)?;
                self.allocation_pool.note_unpooled_node_allocation();
                output
            };
            self.execute(node,&tensors,&output,stream).map_err(|e|format!("{} {}: {e}",node.id,node.op))?;
            if let Some(destination)=diagnostics {
                let bytes=output.download(stream)?;
                let actual=digest(&bytes);
                if actual!=node.output.sha256 {
                    fs::create_dir_all(destination).map_err(|e|e.to_string())?;
                    fs::write(destination.join(format!("{}.raw",node.id)),&bytes).map_err(|e|e.to_string())?;
                    let reference=crate::startup_trace::read(self.root.join(&node.output.file)).map_err(|e|e.to_string())?;
                    let comparison=compare(&bytes,&reference);
                    let report=json!({"passed":false,"node":node.id,"op":node.op,"index":index,"actual_sha256":actual,"gold_sha256":node.output.sha256,"comparison":comparison,"plan":self.plans.get(&node.id)});
                    fs::write(destination.join("first_divergence.json"),serde_json::to_vec_pretty(&report).unwrap()).map_err(|e|e.to_string())?;
                    return Err(format!("first mismatch at {} {}: {comparison}",node.id,node.op));
                }
                if index%100==0{eprintln!("checkpoint {}/{} {} exact",index,nodes.len(),node.id);}
            }
            tensors.insert(node.id.clone(),output);
            for arg in &node.args {if let Some(name)=tensor_id(arg){let count=uses.get_mut(name).ok_or("invalid reference count")?;*count-=1;if *count==0{tensors.remove(name);}}}
        }
        tensors.remove(&self.manifest.output_id).ok_or_else(||"missing final output".into())
    }
    /// Copy final channel-major waveform into the caller's FP32[2,T*800] buffer.
    pub unsafe fn decode(&mut self,input:DevicePtr,output:DevicePtr,stream:DevicePtr)->Result<()> {
        let result=self.run(input,stream,None)?;
        cuda(cudaMemcpyAsync(output,result.pointer(),result.bytes(),3,stream))?;
        cuda(cudaStreamSynchronize(stream))
    }
    unsafe fn execute(&mut self,node:&Node,tensors:&HashMap<String,Tensor>,output:&Tensor,stream:DevicePtr)->Result<()> {
        let get=|v:&Value|->Result<&Tensor>{let id=tensor_id(v).ok_or("expected tensor")?;tensors.get(id).ok_or_else(||format!("missing {id}"))};
        let a=get(&node.args[0])?;
        let count=output.bytes() as i64/4;
        if let Some(result)=self.execute_encoder(node,tensors,output,stream){return result;}
        let shape=shape3(&output.shape)?;
        let null=ptr::null();
        let copy = |base:DevicePtr,as_:[i64;3]|->Result<()>{cuda(h3_audio_pointwise(0,base,null,output.pointer(),count,shape.as_ptr(),as_.as_ptr(),[0i64;3].as_ptr(),0.,0.,stream))};
        match node.op.as_str() {
            "aten.convolution.default"=>{
                let w=get(&node.args[1])?;
                let b=if node.args[2].is_null(){None}else{Some(get(&node.args[2])?)};
                let stride=first(&node.args[3])?;
                let pad=first(&node.args[4])?;
                let dilation=first(&node.args[5])?;
                let transposed=node.args[6].as_bool().ok_or("missing transposed flag")?;
                let group=integer(&node.args[8])?;
                if !transposed&&group==a.shape[1]&&group>1 {
                    if w.shape[0]!=group||w.shape[1]!=1{return Err("unsupported depth multiplier".into());}
                    cuda(h3_audio_depthwise(a.pointer(),w.pointer(),b.map_or(null,|v|v.pointer().cast_const()),output.pointer(),count as i32,group as i32,a.shape[2] as i32,output.shape[2] as i32,w.shape[2] as i32,stride as i32,pad as i32,dilation as i32,stream))?;
                    self.plans.insert(node.id.clone(),"original-torch-fp32-depthwise-generic".into());
                }else{
                    let spec=ConvSpec{ndim:4,dtype:0,transposed:transposed as i32,
                        allow_tf32:self.manifest.provenance["cudnn_allow_tf32"].as_bool().unwrap_or(false) as i32,
                        deterministic:self.manifest.provenance["cudnn_deterministic"].as_bool().unwrap_or(false) as i32,
                        input_dims:dims4(&a.shape),input_strides:stride4(&a.shape),weight_dims:dims4(&w.shape),weight_strides:stride4(&w.shape),output_dims:dims4(&output.shape),output_strides:stride4(&output.shape),
                        padding:[0,pad,0],stride:[1,stride,0],dilation:[1,dilation,0],..Default::default()};
                    conv(h3_conv_forward(self.context,&spec,a.pointer(),w.pointer(),output.pointer(),stream))?;
                    self.plans.insert(node.id.clone(),CStr::from_ptr(h3_conv_plan_tag(self.context)).to_string_lossy().into_owned());
                    if let Some(bias)=b {cuda(h3_audio_pointwise(2,output.pointer(),bias.pointer(),output.pointer(),count,shape.as_ptr(),[shape[1]*shape[2],shape[2],1].as_ptr(),[0i64,1,0].as_ptr(),0.,0.,stream))?;}
                }
                Ok(())
            },
            "aten.replication_pad1d.default"=>cuda(h3_audio_pad(a.pointer(),output.pointer(),count,a.shape[2] as i32,output.shape[2] as i32,first(&node.args[1])? as i32,stream)),
            "aten.slice.Tensor"=>{
                let rank=a.shape.len() as i64;
                let mut dim=integer(&node.args[1])?;if dim<0{dim+=rank;}
                let start=integer(&node.args[2])?;let start=if start<0{start+a.shape[dim as usize]}else{start};
                let step=if node.args.len()>4{integer(&node.args[4])?}else{1};
                let mut ss=strides(&a.shape);let offset=start*ss[dim as usize];ss[dim as usize]*=step;
                let mut s=[0;3];s[3-ss.len()..].copy_from_slice(&ss);
                copy(a.pointer().cast::<f32>().offset(offset as isize).cast(),s)
            },
            "aten.permute.default"=>{
                let order=list(&node.args[1])?;let ss=strides(&a.shape);let mut s=[0;3];
                for(i,d)in order.iter().enumerate(){s[3-order.len()+i]=ss[*d as usize];}copy(a.pointer(),s)
            },
            "aten.clone.default"|"aten._to_copy.default"|"aten.detach.default"|"aten.alias.default"|"aten.view.default"|"aten.reshape.default"|"aten.unsqueeze.default"|"aten.squeeze.dim"=>copy(a.pointer(),[shape[1]*shape[2],shape[2],1]),
            "aten.mul.Tensor"|"aten.mul.Scalar"|"aten.add.Tensor"|"aten.add.Scalar"|"aten.div.Tensor"|"aten.div.Scalar"|"aten.sin.default"|"aten.pow.Tensor_Scalar"|"aten.clamp.default"=>{
                let as_=broadcast(a,&output.shape)?;let mut bs=[0;3];let mut bptr=null;let mut scalar=0f32;
                let op=match node.op.as_str(){
                    "aten.sin.default"=>3,
                    "aten.pow.Tensor_Scalar"=>{if node.args[1].as_f64()!=Some(2.){return Err("unsupported power".into());}4},
                    "aten.clamp.default"=>{if node.args[1].as_f64()!=Some(-1.)||node.args[2].as_f64()!=Some(1.){return Err("unsupported clamp".into());}5},
                    _=>{
                        if tensor_id(&node.args[1]).is_some(){let b=get(&node.args[1])?;bptr=b.pointer().cast_const();bs=broadcast(b,&output.shape)?;}
                        else{scalar=node.args[1].as_f64().ok_or("expected scalar")? as f32;}
                        if node.op.starts_with("aten.div"){
                            if !bptr.is_null(){return Err("dynamic tensor division not supported".into());}
                            scalar=(1.0f64/node.args[1].as_f64().unwrap()) as f32;1
                        }else if node.op.starts_with("aten.add"){
                            if let Some(alpha)=node.kwargs.get("alpha"){if alpha.as_f64()!=Some(1.){return Err("unsupported add alpha".into());}}2
                        }else{1}
                    }
                };
                cuda(h3_audio_pointwise(op,a.pointer(),bptr,output.pointer(),count,shape.as_ptr(),as_.as_ptr(),bs.as_ptr(),0.,scalar,stream))
            },
            other=>Err(format!("unsupported operation {other}")),
        }
    }
}

pub fn compare(actual:&[u8],gold:&[u8])->Value {
    let mut mismatches=0usize;let mut nonfinite=0usize;let mut max_abs=0f64;let mut first=None;
    for (index,(a,b)) in actual.chunks_exact(4).zip(gold.chunks_exact(4)).enumerate(){
        let x=f32::from_le_bytes(a.try_into().unwrap());let y=f32::from_le_bytes(b.try_into().unwrap());
        if !x.is_finite(){nonfinite+=1;}
        if a!=b{mismatches+=1;if first.is_none(){first=Some(json!({"index":index,"actual":x,"gold":y}));}max_abs=max_abs.max((x as f64-y as f64).abs());}
    }
    json!({"elements":actual.len()/4,"mismatches":mismatches,"nonfinite":nonfinite,"max_abs":max_abs,"first":first,"bytes_equal":actual==gold})
}
