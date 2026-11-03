use crate::{prepare::hash, Result};
use serde_json::{json, Value};
use std::{ffi::{c_char, c_int, c_void, CStr}, fs, path::Path, ptr};
mod model;
pub use model::infer_with_size;
pub use model::infer_ref2va;

type Stream = *mut c_void;
unsafe extern "C" {
    fn cudaGetErrorString(code: c_int) -> *const c_char;
    fn cudaMalloc(pointer: *mut *mut c_void, bytes: usize) -> c_int;
    fn cudaFree(pointer: *mut c_void) -> c_int;
    fn cudaHostAlloc(pointer: *mut *mut c_void, bytes: usize, flags: u32) -> c_int;
    fn cudaFreeHost(pointer: *mut c_void) -> c_int;
    fn cudaStreamCreateWithFlags(stream: *mut Stream, flags: u32) -> c_int;
    fn cudaStreamSynchronize(stream: Stream) -> c_int;
    fn cudaStreamDestroy(stream: Stream) -> c_int;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int, stream: Stream) -> c_int;
    fn h3_linear_create(stream: Stream, workspace: *mut c_void, bytes: usize, lt_workspace: *mut c_void, lt_bytes: usize,
        tf32: c_int, reduced: c_int, out: *mut *mut c_void) -> c_int;
    fn h3_linear_forward(handle: *mut c_void, x: *const c_void, w: *const c_void, bias: *const c_void,
        out: *mut c_void, rows: c_int, n: c_int, k: c_int, dtype: c_int) -> c_int;
    fn h3_linear_destroy(handle: *mut c_void);
    fn h3_linear_prepare_qwen(handle:*mut c_void)->c_int;
    fn h3_linear_error() -> *const c_char;
    fn q3_rms(x: *const c_void, w: *const c_void, out: *mut c_void, rows: i64, cols: c_int, eps: f32, stream: Stream) -> c_int;
    fn q3_layernorm(x: *const c_void, w: *const c_void, bias: *const c_void, out: *mut c_void, rows: i64, cols: c_int, eps: f32, stream: Stream) -> c_int;
    fn q3_activation(x: *const c_void, out: *mut c_void, n: i64, kind: c_int, stream: Stream) -> c_int;
    fn q3_binary(x: *const c_void, y: *const c_void, out: *mut c_void, n: i64, kind: c_int, ycols: c_int, stream: Stream) -> c_int;
    fn q3_attention_workspace(kind: c_int, sequence: i64, bytes: *mut usize) -> c_int;
    fn q3_attention_forward(kind: c_int, sequence: i64, q: *const c_void, k: *const c_void, v: *const c_void, out: *mut c_void,
        workspace: *mut c_void, bytes: usize, scale: f32, stream: Stream) -> c_int;
    fn q3_attention_plan_name(kind: c_int, sequence: i64) -> *const c_char;
    fn q3_attention_error() -> *const c_char;
    fn q3_attention_cleanup() -> c_int;
}

fn cuda(code: c_int) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(format!("CUDA {code}: {}",unsafe {CStr::from_ptr(cudaGetErrorString(code))}.to_string_lossy()).into()) }
}
fn native_error(code: c_int, error: unsafe extern "C" fn() -> *const c_char) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(unsafe {CStr::from_ptr(error())}.to_string_lossy().into_owned().into()) }
}
#[derive(Clone, Copy)]
struct Buffer { ptr: *mut c_void, bytes: usize }
struct Arena { stream: Stream, allocations: Vec<Buffer>, linear: *mut c_void }
impl Arena {
    fn new() -> Result<Self> {
        let _trace=crate::startup_trace::Span::new("cuda_context","qwen-arena",0);
        let mut stream=ptr::null_mut();
        crate::startup_trace::measure("qwen_cuda_stage","stream",0,||cuda(unsafe {cudaStreamCreateWithFlags(&mut stream,1)}))?;
        let mut arena=Self{stream,allocations:vec![],linear:ptr::null_mut()};
        let (blas,lt)=crate::startup_trace::measure("qwen_cuda_stage","workspaces",33<<20,||->Result<_> {Ok((arena.alloc(32<<20)?,arena.alloc(1<<20)?))})?;
        let _backend=crate::startup_trace::Span::new("qwen_backend_init","owner",0);
        crate::startup_trace::measure("qwen_cuda_stage","linear-create",0,||native_error(unsafe{h3_linear_create(stream,blas.ptr,blas.bytes,lt.ptr,lt.bytes,0,1,&mut arena.linear)},h3_linear_error))?;
        if crate::weight_loader::flag("H3_QWEN_BACKEND_PREPARE")? {
            let code=crate::startup_trace::measure("qwen_cuda_stage","backend-metadata",0,||unsafe{h3_linear_prepare_qwen(arena.linear)});
            if code!=0{eprintln!("Qwen backend preparation deferred: {code}");}
        }
        Ok(arena)
    }
    fn alloc(&mut self, bytes: usize) -> Result<Buffer> {
        model::DYNAMIC_ALLOCATIONS.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
        let mut pointer=ptr::null_mut();
        cuda(unsafe {cudaMalloc(&mut pointer,bytes.max(1))})?;
        let b=Buffer{ptr:pointer,bytes};self.allocations.push(b);Ok(b)
    }
    fn sync(&self) -> Result<()> {cuda(unsafe {cudaStreamSynchronize(self.stream)})}
    fn alloc_model_tensors(&mut self,sizes:&[usize])->Result<Vec<Buffer>> {
        if sizes.is_empty() || !crate::weight_loader::flag("H3_MODEL_LOAD_SLAB")? {
            return sizes.iter().map(|&bytes|self.alloc(bytes)).collect();
        }
        let (offsets,bytes)=crate::weight_loader::tensor_allocation_layout(sizes)?;
        let owned=self.alloc(bytes)?;
        // Only `owned` is freed by the arena. The returned views never own or
        // free a slice of this allocation, including on a failed upload.
        Ok(offsets.iter().zip(sizes).map(|(&offset,&bytes)|Buffer {
            ptr:unsafe{owned.ptr.cast::<u8>().add(offset).cast()},bytes,
        }).collect())
    }
    fn upload(&mut self, bytes: &[u8]) -> Result<Buffer> {
        let out=self.alloc(bytes.len())?;
        let result=cuda(unsafe {cudaMemcpyAsync(out.ptr,bytes.as_ptr().cast(),bytes.len(),1,self.stream)});
        let synced=self.sync(); result?;synced?;Ok(out)
    }
    fn read(&self, buffer:Buffer) -> Result<Vec<u8>> {
        let mut output=vec![0;buffer.bytes];
        let result=cuda(unsafe {cudaMemcpyAsync(output.as_mut_ptr().cast(),buffer.ptr,buffer.bytes,2,self.stream)});
        let synced=self.sync();result?;synced?;Ok(output)
    }
}
impl Drop for Arena {
    fn drop(&mut self) {
        let _=self.sync();
        if !self.linear.is_null() {unsafe {h3_linear_destroy(self.linear)}};
        for b in &self.allocations {if !b.ptr.is_null(){unsafe {cudaFree(b.ptr);}}}
        unsafe {cudaStreamDestroy(self.stream);}
    }
}

fn shape(spec: &Value) -> Result<Vec<usize>> {
    spec["shape"].as_array().ok_or("tensor shape")?.iter().map(|x| x.as_u64().map(|v|v as usize).ok_or_else(||"shape dimension".into())).collect()
}
fn raw(root: &Path, spec: &Value) -> Result<Vec<u8>> {
    crate::packed_weights::read(root,spec)
}
fn compare(a:&[u8],b:&[u8]) -> Result<Value> {
    if a.len()!=b.len() || a.is_empty() || a.len()%2!=0 {return Err("BF16 comparison size".into());}
    let (mut different,mut nonfinite,mut peak,mut sum,mut first)=(0usize,0usize,0f64,0f64,None);
    for (i,(a,b)) in a.chunks_exact(2).zip(b.chunks_exact(2)).enumerate() {
        let x=u16::from_le_bytes(a.try_into().unwrap());let y=u16::from_le_bytes(b.try_into().unwrap());
        if x!=y {different+=1;if first.is_none(){first=Some(i);}}
        let xf=f32::from_bits((x as u32)<<16) as f64;let yf=f32::from_bits((y as u32)<<16) as f64;
        if !xf.is_finite() || !yf.is_finite(){nonfinite+=1;continue;}
        let delta=(xf-yf).abs();peak=peak.max(delta);sum+=delta;
    }
    Ok(json!({"passed":different==0&&nonfinite==0,"elements":a.len()/2,"mismatches":different,"nonfinite":nonfinite,
              "first_mismatch":first,"max_abs":peak,"mean_abs":sum/(a.len()/2) as f64,"sha256":hash(a)}))
}
fn audit(out:&Path,phase:&str) -> Result<()> {
    if !crate::debug_audit::enabled(){return Ok(());}
    let maps=fs::read_to_string("/proc/self/maps")?;
    fs::write(out.join(format!("native.{phase}.maps.txt")),&maps)?;
    if ["libtorch","libpython","libaten","libc10"].iter().any(|s|maps.to_lowercase().contains(s)){return Err("forbidden native dependency".into());}
    Ok(())
}
fn input_spec(fixture:&Value) -> Result<&Value> {
    fixture["inputs"].as_array().and_then(|x|x.first()).or_else(||fixture["kwargs"].get("hidden_states")).ok_or_else(||"fixture input".into())
}

fn replay_module(root:&Path,manifest:&Value,name:&str) -> Result<Value> {
    let fixture=&manifest["module_fixtures"][name];
    let kind=manifest["modules"][name]["type"].as_str().ok_or("module type")?;
    let input=input_spec(fixture)?;let expected=&fixture["output"];
    if input["dtype"]!="torch.bfloat16" || expected["dtype"]!="torch.bfloat16" {return Err(format!("unsupported module dtype: {name}").into());}
    let input_shape=shape(input)?;let elements:usize=input_shape.iter().product();
    let mut arena=Arena::new()?;
    let x=arena.upload(&raw(root,input)?)?;let gold=raw(root,expected)?;let output=arena.alloc(gold.len())?;
    let weight_spec=&manifest["weights"][format!("{name}.weight")];
    let bias_spec=&manifest["weights"][format!("{name}.bias")];
    let weight=if weight_spec.is_null(){None}else{Some(arena.upload(&raw(root,weight_spec)?)?)};
    let bias=if bias_spec.is_null(){None}else{Some(arena.upload(&raw(root,bias_spec)?)?)};
    let mut runs=vec![];
    for repeat in 0..3 {
        let cols=*input_shape.last().ok_or("input width")?;
        let rows=elements/cols;
        if kind.ends_with(".Linear") {
            let ws=shape(weight_spec)?;
            native_error(unsafe {h3_linear_forward(arena.linear,x.ptr,weight.unwrap().ptr,
                bias.map_or(ptr::null_mut(),|b|b.ptr),output.ptr,rows as i32,ws[0] as i32,ws[1] as i32,0)},h3_linear_error)?;
        } else if kind.ends_with(".LayerNorm") {
            cuda(unsafe {q3_layernorm(x.ptr,weight.unwrap().ptr,bias.unwrap().ptr,output.ptr,rows as i64,cols as i32,1e-6,arena.stream)})?;
        } else if kind.ends_with(".Qwen3VLTextRMSNorm") {
            cuda(unsafe {q3_rms(x.ptr,weight.unwrap().ptr,output.ptr,rows as i64,cols as i32,1e-6,arena.stream)})?;
        } else {
            let activation=if kind.ends_with(".SiLUActivation"){0}else if kind.ends_with(".GELUTanh"){1}else if kind.ends_with(".GELU"){2}else{return Err("unsupported activation".into());};
            cuda(unsafe {q3_activation(x.ptr,output.ptr,elements as i64,activation,arena.stream)})?;
        }
        let actual=arena.read(output)?;let result=compare(&actual,&gold)?;
        runs.push(json!({"repeat":repeat,"comparison":result}));
    }
    Ok(json!({"name":name,"kind":kind,"passed":runs.iter().all(|v|v["comparison"]["passed"]==true),"replays":runs}))
}

fn storage_offsets(spec:&Value) -> Result<(Vec<usize>,usize)> {
    let dims=shape(spec)?;
    let strides:Vec<usize>=spec["source"]["stride"].as_array().ok_or("source strides")?.iter().map(|x|x.as_u64().unwrap() as usize).collect();
    let mut offsets=Vec::with_capacity(dims.iter().product());
    for mut index in 0..dims.iter().product::<usize>() {
        let mut offset=0;
        for dimension in (0..dims.len()).rev(){offset+=(index%dims[dimension])*strides[dimension];index/=dims[dimension];}
        offsets.push(offset);
    }
    let size=offsets.iter().max().ok_or("empty tensor")?+1;Ok((offsets,size))
}
fn strided(root:&Path,spec:&Value) -> Result<Vec<u8>> {
    let source=raw(root,spec)?;let (offsets,size)=storage_offsets(spec)?;let mut result=vec![0;size*2];
    for (i,offset) in offsets.into_iter().enumerate(){result[offset*2..offset*2+2].copy_from_slice(&source[i*2..i*2+2]);}
    Ok(result)
}
fn replay_attention(root:&Path,call:&Value,kind:i32) -> Result<Value> {
    let fixture=&call["fixture"];let mut arena=Arena::new()?;
    let q=arena.upload(&strided(root,&fixture["q"])?)?;
    let k=arena.upload(&strided(root,&fixture["k"])?)?;
    let v=arena.upload(&strided(root,&fixture["v"])?)?;
    let (offsets,size)=storage_offsets(&fixture["output"])?;
    let output=arena.alloc(size*2)?;let gold=raw(root,&fixture["output"])?;
    let sequence=shape(&fixture["q"])?[2] as i64;
    let mut bytes=0;
    native_error(unsafe {q3_attention_workspace(kind,sequence,&mut bytes)},q3_attention_error)?;
    let workspace=arena.alloc(bytes)?;
    let scale=call["kwargs"]["scale"].as_f64().ok_or("attention scale")? as f32;
    let mut runs=vec![];
    for repeat in 0..3 {
        native_error(unsafe {q3_attention_forward(kind,sequence,q.ptr,k.ptr,v.ptr,output.ptr,workspace.ptr,bytes,scale,arena.stream)},q3_attention_error)?;
        let actual=arena.read(output)?;let contiguous:Vec<u8>=offsets.iter().flat_map(|&offset|[actual[offset*2],actual[offset*2+1]]).collect();
        runs.push(json!({"repeat":repeat,"comparison":compare(&contiguous,&gold)?}));
    }
    let plan=unsafe {CStr::from_ptr(q3_attention_plan_name(kind,sequence))}.to_string_lossy().into_owned();
    Ok(json!({"name":if kind==0{"vision_sdpa"}else{"text_sdpa"},"passed":runs.iter().all(|v|v["comparison"]["passed"]==true),"plan_name":plan,"workspace_bytes":bytes,"replays":runs}))
}

pub fn replay_ops(root:&Path,output:&Path) -> Result<Value> {
    fs::create_dir_all(output)?;audit(output,"start")?;
    let manifest:Value=serde_json::from_slice(&crate::startup_trace::read(root.join("manifest.json"))?)?;
    if manifest["status"]!="complete" || manifest["schema"]!="h3-conditioner-export-v1" {return Err("incomplete conditioner export".into());}
    let mut results=vec![];
    for (name,_) in manifest["module_fixtures"].as_object().ok_or("fixtures")? {
        let kind=manifest["modules"][name]["type"].as_str().unwrap_or("");
        if ![".Linear",".LayerNorm",".Qwen3VLTextRMSNorm",".SiLUActivation",".GELUTanh",".GELU"].iter().any(|suffix|kind.ends_with(suffix)){continue;}
        let result=replay_module(root,&manifest,name)?;
        eprintln!("{}",serde_json::to_string(&result)?);results.push(result);
        fs::write(output.join("ops.partial.json"),serde_json::to_vec_pretty(&results)?)?;
    }
    for call in manifest["attention_calls"].as_array().ok_or("attention calls")? {
        if call.get("fixture").is_none(){continue;}
        let kind=if call["scope"].as_str().unwrap().starts_with("visual"){0}else{1};
        let result=replay_attention(root,call,kind)?;
        eprintln!("{}",serde_json::to_string(&result)?);results.push(result);
    }
    native_error(unsafe{q3_attention_cleanup()},q3_attention_error)?;
    audit(output,"complete")?;
    let report=json!({"schema":"h3-conditioner-op-replay-v1","passed":results.iter().all(|x|x["passed"]==true),
        "fixture_manifest_sha256":hash(&crate::startup_trace::read(root.join("manifest.json"))?),"native_process_clean":true,
        "repeats_per_fixture":3,"results":results});
    fs::write(output.join("ops-report.json"),serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub use model::{serve, write_unique_id};
