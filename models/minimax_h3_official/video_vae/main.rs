//! Native execution of the original compiled VAE kernel graph.
//! The manifest supplies code/weights/layout, never downstream activations.
#[path="../support/startup_trace.rs"]
pub mod startup_trace;
#[path="../support/debug_audit.rs"]
pub mod debug_audit;
#[path="../support/limits.rs"]
pub mod limits;
use std::{collections::HashMap,ffi::{c_char,c_int,c_uint,c_void,CStr,CString},fs,
          io::{Read},path::{Path,PathBuf},ptr,time::Instant};
use h3_router::{Arena,DevicePtr,Stream,check};
use serde_json::{Value,json};
use sha2::{Digest,Sha256};
#[path="../support/weight_loader.rs"] mod weight_loader;
mod full;
#[path="../support/light_sha.rs"] mod light_sha;
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
#[repr(C)]
#[derive(Default)]
struct ConvSpec { ndim:i32,dtype:i32,transposed:i32,allow_tf32:i32,deterministic:i32,channels_last:i32,
    input_dims:[i64;5],input_strides:[i64;5],weight_dims:[i64;5],weight_strides:[i64;5],
    output_dims:[i64;5],output_strides:[i64;5],padding:[i64;3],stride:[i64;3],dilation:[i64;3] }
unsafe extern "C" {
    fn cuModuleLoadData(module:*mut *mut c_void,image:*const c_void)->c_int;
    fn cuModuleUnload(module:*mut c_void)->c_int;
    fn cuModuleGetFunction(function:*mut *mut c_void,module:*mut c_void,name:*const c_char)->c_int;
    fn cuFuncSetAttribute(function:*mut c_void,attribute:c_int,value:c_int)->c_int;
    fn cuFuncGetParamInfo(function:*mut c_void,index:usize,offset:*mut usize,size:*mut usize)->c_int;
    fn cuLaunchKernel(function:*mut c_void,gx:c_uint,gy:c_uint,gz:c_uint,bx:c_uint,by:c_uint,bz:c_uint,
        shared:c_uint,stream:Stream,params:*mut *mut c_void,extra:*mut *mut c_void)->c_int;
    fn cuMemcpyDtoDAsync_v2(dst:DevicePtr,src:DevicePtr,bytes:usize,stream:Stream)->c_int;
    fn h3_linear_create(stream:*mut c_void,blas_ws:*mut c_void,blas_bytes:usize,lt_ws:*mut c_void,
        lt_bytes:usize,allow_tf32:c_int,allow_reduced:c_int,out:*mut *mut c_void)->c_int;
    fn h3_linear_forward(ctx:*mut c_void,input:*const c_void,weight:*const c_void,bias:*const c_void,
        output:*mut c_void,rows:c_int,out_features:c_int,in_features:c_int,dtype:c_int)->c_int;
    fn h3_linear_error()->*const c_char;
    fn h3v_mm(ctx:*mut c_void,input:*const c_void,weight:*const c_void,output:*mut c_void,
        rows:c_int,out_features:c_int,in_features:c_int,input_column_major:c_int)->c_int;
    fn h3_linear_destroy(ctx:*mut c_void);
    fn h3v_dense(q:*const c_void,k:*const c_void,v:*const c_void,out:*mut c_void,dims:*const i64,
        qs:*const i64,ks:*const i64,vs:*const i64,os:*const i64,scale:f32,stream:*mut c_void)->c_int;
    fn h3v_dense_error()->*const c_char;
    fn h3v_dense_cleanup()->c_int;
    fn h3_conv_create(ctx:*mut *mut c_void)->c_int;
    fn h3_conv_prepare(ctx:*mut c_void,spec:*const ConvSpec,x:*const c_void,w:*const c_void,y:*mut c_void,stream:*mut c_void)->i32;
    fn h3v_dense_prepare(dims:*const i64,qs:*const i64,ks:*const i64,vs:*const i64,os:*const i64)->i32;
    fn h3_conv_forward(ctx:*mut c_void,spec:*const ConvSpec,x:*const c_void,w:*const c_void,y:*mut c_void,stream:*mut c_void)->c_int;
    fn h3_conv_error()->*const c_char;
    fn h3_conv_destroy(ctx:*mut c_void)->c_int;
    fn h3v_pad(x:*const c_void,y:*mut c_void,shape:*const i64,stride:*const i64,out:*const i64,pad:*const i64,reflect:i32,stream:*mut c_void)->i32;
    fn h3v_group_norm(x:*const c_void,gamma:*const c_void,beta:*const c_void,y:*mut c_void,mean:*mut c_void,rstd:*mut c_void,scratch:*mut c_void,n:i64,c:i64,spatial:i64,groups:i64,eps:f32,stream:*mut c_void)->i32;
    fn h3v_pointwise(x:*const c_void,z:*const c_void,y:*mut c_void,count:i64,op:i32,spatial:i64,channels:i64,stream:*mut c_void)->i32;
}
fn num(v:&Value)->Result<usize>{Ok(v.as_u64().ok_or("expected unsigned integer")? as usize)}
fn text(v:&Value)->Result<&str>{Ok(v.as_str().ok_or("expected string")?)}
fn arr(v:&Value)->Result<&Vec<Value>>{Ok(v.as_array().ok_or("expected array")?)}
fn ints(v:&Value)->Result<Vec<i64>>{arr(v)?.iter().map(|x|x.as_i64().ok_or_else(||"expected integer".into())).collect()}
fn hash(bytes:&[u8])->String{format!("{:x}",crate::startup_trace::measure("sha256","",(bytes).len(),||Sha256::digest(bytes)))}
fn read_checked(root:&Path,spec:&Value)->Result<Vec<u8>>{
    let path=root.join(text(&spec["file"])?);let mut source=fs::File::open(&path)?;let mut bytes=Vec::new();crate::startup_trace::measure("read_file",path.to_string_lossy(),num(&spec["bytes"])? ,||source.read_to_end(&mut bytes))?;
    if bytes.len()!=num(&spec["bytes"])? || hash(&bytes)!=text(&spec["sha256"])? {return Err(format!("tensor identity mismatch: {}",path.display()).into());}
    Ok(bytes)
}
fn read_model_checked(root:&Path,spec:&Value)->Result<Vec<u8>>{
    let path=root.join(text(&spec["file"])?);let bytes=crate::startup_trace::read(&path)?;
    if bytes.len()!=num(&spec["bytes"])? {return Err(format!("model tensor length mismatch: {}",path.display()).into());}
    light_sha::verify_model_bytes(&bytes,text(&spec["sha256"])?)?;Ok(bytes)
}
struct Kernel{module:*mut c_void,function:*mut c_void,threads:u32,shared:u32,params:usize}
impl Drop for Kernel{fn drop(&mut self){unsafe{cuModuleUnload(self.module);}}}
struct Program{spatial:SpatialPlans,arena:Arena,linear:*mut c_void,conv:*mut c_void,norm_scratch:DevicePtr,manifest:std::sync::Arc<Value>,buffers:Vec<DevicePtr>,kernels:HashMap<String,Kernel>,workspace_layout:Vec<(usize,usize)>,workspace_bytes:usize}
include!("decoder_manifest.rs");
include!("encoder_manifest.rs");
include!("spatial_plans.rs");
include!("weight_casts.rs");
unsafe extern "C" {
    fn h3_conv_cache_stats(ctx:*mut c_void,out:*mut u64)->i32;
    fn h3_conv_release_workspace(ctx:*mut c_void)->i32;
    fn h3v_dense_cache_stats(out:*mut u64)->i32;
    fn h3v_dense_workspace_bytes()->u64;
    fn h3v_dense_release_workspace()->i32;
    fn h3v_generic_decoder_kernel(kind:i32,params:*const u64,count:i32,t:i32,h:i32,w:i32,stream:*mut c_void)->i32;
}
fn release_conv_workspace(ctx:*mut c_void)->Result<Value>{
    let(mut before,mut after)=([0u64;9],[0u64;9]);
    unsafe{h3_conv_cache_stats(ctx,before.as_mut_ptr());}
    if unsafe{h3_conv_release_workspace(ctx)}!=0{return Err(unsafe{CStr::from_ptr(h3_conv_error())}.to_string_lossy().into_owned().into());}
    unsafe{h3_conv_cache_stats(ctx,after.as_mut_ptr());}
    if after[5]!=0 || before[..5]!=after[..5]{return Err("idle cuDNN release changed cached convolution plans".into());}
    Ok(json!({"released_bytes":before[5],"workspace_bytes":after[5],"plan_entries":after[0],"plans_preserved":true,"sync_before_free":true}))
}
fn release_dense_workspace()->Result<Value>{
    let(mut before,mut after)=([0u64;5],[0u64;5]);
    unsafe{h3v_dense_cache_stats(before.as_mut_ptr());}
    let released=unsafe{h3v_dense_workspace_bytes()};
    if unsafe{h3v_dense_release_workspace()}!=0{return Err(unsafe{CStr::from_ptr(h3v_dense_error())}.to_string_lossy().into_owned().into());}
    unsafe{h3v_dense_cache_stats(after.as_mut_ptr());}
    let remaining=unsafe{h3v_dense_workspace_bytes()};
    if remaining!=0 || before!=after{return Err("idle SDPA release changed cached attention plans".into());}
    Ok(json!({"released_bytes":released,"workspace_bytes":remaining,"plan_entries":after[0],"plans_preserved":true,"sync_before_free":true}))
}
impl Program{
    fn ensure_workspace(&mut self)->Result<()>{
        if self.workspace_layout.first().is_some_and(|&(id,_)|self.buffers[id]==0){
            let base=self.arena.alloc(self.workspace_bytes)?;
            for &(id,offset) in &self.workspace_layout{self.buffers[id]=base+offset as u64;}
        }
        Ok(())
    }
    // Called only after the owning Context has synchronized and released every
    // allocation following its persistent parameter/handle checkpoint.
    fn workspace_released(&mut self){for &(id,_) in &self.workspace_layout{self.buffers[id]=0;}}
    fn select_spatial(&mut self,stage:&str,h:usize,w:usize)->Result<()>{
        self.arena.sync()?;
        self.ensure_workspace()?;
        self.manifest=self.spatial.select(stage,h,w)?;
        Ok(())
    }
    fn spatial_report(&self)->Value{
        let mut report=self.spatial.report();let mut conv=[0u64;9];let mut dense=[0u64;5];
        unsafe{h3_conv_cache_stats(self.conv,conv.as_mut_ptr());h3v_dense_cache_stats(dense.as_mut_ptr());}
        report["conv_cache"]=json!({"entries":conv[0],"capacity":conv[1],"hits":conv[2],"misses":conv[3],"evictions":conv[4],"workspace_bytes":conv[5],"workspace_limit":conv[6],"workspace_allocations":conv[7],"oversized_executions":conv[8]});
        report["dense_cache"]=json!({"entries":dense[0],"capacity":dense[1],"hits":dense[2],"builds":dense[3],"evictions":dense[4],"workspace_bytes":unsafe{h3v_dense_workspace_bytes()}});
        report["request_workspace_bytes"]=json!(self.workspace_bytes);
        report["workspace_released"]=json!(self.workspace_layout.first().is_some_and(|&(id,_)|self.buffers[id]==0));
        if !self.workspace_layout.is_empty(){report["allocation_policy"]=json!("request-scoped maximum-tile slab; parameters and plans remain resident");}report
    }

    fn load(root:&Path)->Result<Self>{
        Self::load_execution(root,true)
    }
    fn load_execution(root:&Path,initialize_inputs:bool)->Result<Self>{
        Self::load_execution_options(root,initialize_inputs,false)
    }
    fn load_execution_options(root:&Path,initialize_inputs:bool,read_ahead:bool)->Result<Self>{
        let started=Instant::now();let manifest:Value=serde_json::from_slice(&crate::startup_trace::read(root.join("program.json"))?)?;
        if !["h3_original_inductor_v1","h3_original_eager_v1"].contains(&text(&manifest["schema"])?) || manifest["no_intermediate_initializers"]!=true{return Err("unsupported or unsafe native program manifest".into());}
        let audit=PathBuf::from(text(&manifest["audit_root"])?);
        if hash(&crate::startup_trace::read(audit.join("manifest.json"))?)!=text(&manifest["audit_manifest_sha256"])?{return Err("VAE audit manifest identity mismatch".into());}
        let backend=&manifest["backend"];
        if backend["allow_tf32"]!=false || backend["allow_fp16_reduced_precision_reduction"]!=true{return Err("unvalidated VAE GEMM backend".into());}
        let mut prefetched=if (read_ahead || crate::weight_loader::flag("H3_MODEL_LOAD_INIT_PREFETCH")?) && std::env::var("H3_VAE_LOAD_PIPELINE").as_deref()==Ok("1"){
            let config=crate::weight_loader::Config::from_env()?.ok_or("VAE pipeline requires model pipeline")?;
            let mut jobs=Vec::new();
            for spec in arr(&manifest["buffers"])?{
                if spec["role"]=="parameter" || (spec["role"]=="input" && initialize_inputs){
                    let s=&spec["source"]["spec"];
                    jobs.push(crate::weight_loader::Job::new(&audit,text(&s["file"])?,num(&s["bytes"])?,Some(text(&s["sha256"])?.to_owned()))?);
                }
            }
            Some(if read_ahead {crate::weight_loader::ReaderPool::start_read_ahead(jobs,config)?}else{crate::weight_loader::ReaderPool::start(jobs,config)?})
        }else{None};
        let mut arena=Arena::new()?;let mut buffers=Vec::new();
        let defer_workspace=!initialize_inputs && std::env::var("H3_VAE_RELEASE_IDLE_BUFFERS").as_deref()==Ok("1");
        let mut workspace_layout=Vec::new();let mut workspace_bytes=0usize;
        let specs=arr(&manifest["buffers"])?;
        if std::env::var("H3_VAE_LOAD_PIPELINE").as_deref()==Ok("1") {
            let config=crate::weight_loader::Config::from_env()?.ok_or("VAE pipeline requires model pipeline")?;
            let mut jobs=Vec::new();let mut destinations=Vec::new();
            for spec in specs {
                let bytes=num(&spec["bytes"])?;
                if num(&spec["id"])?!=buffers.len(){return Err("noncanonical allocation ids".into());}
                let pointer=if defer_workspace && spec["role"]!="parameter" {
                    workspace_layout.push((buffers.len(),workspace_bytes));
                    workspace_bytes=workspace_bytes.checked_add(bytes.max(8)).and_then(|v|v.checked_add(255)).ok_or("VAE workspace size overflow")?/256*256;0
                }else{arena.alloc(bytes.max(8))?};
                match spec["role"].as_str() {
                    Some("parameter")|Some("input")=>{
                        if spec["role"]=="parameter" || initialize_inputs {
                            let s=&spec["source"]["spec"];let n=num(&s["bytes"])?;
                            if n>bytes{return Err("initializer exceeds buffer".into());}
                            jobs.push(crate::weight_loader::Job::new(&audit,text(&s["file"])?,n,Some(text(&s["sha256"])?.to_owned()))?);
                            destinations.push(pointer);
                        }
                    },
                    Some("scratch")=>{if !spec["source"].is_null(){return Err("scratch buffer cannot have an initializer".into());}},
                    _=>return Err("unknown buffer role".into()),
                }
                buffers.push(pointer);
            }
            let mut pool=if let Some(pool)=prefetched.take(){pool}else{crate::weight_loader::ReaderPool::start(jobs,config)?};
            let mut upload=crate::weight_loader::device::Upload::new(arena.stream,config)?;
            unsafe{pool.consume_upload(&mut upload,|i,offset|(destinations[i] as usize+offset) as *mut c_void)}?;
            upload.finish()?;
        }else {
        for chunk in specs.chunks(8){
            let contents=std::thread::scope(|scope|{
                let jobs:Vec<_>=chunk.iter().map(|spec|{let audit=&audit;scope.spawn(move||->std::result::Result<Option<Vec<u8>>,String>{
                    match spec["role"].as_str(){Some("parameter")=>read_model_checked(audit,&spec["source"]["spec"]).map(Some).map_err(|e|e.to_string()),
                    Some("input")=>if initialize_inputs{read_checked(audit,&spec["source"]["spec"]).map(Some).map_err(|e|e.to_string())}else{Ok(None)},
                    Some("scratch")=>Ok(None),_=>Err("unknown buffer role".into())}
                })}).collect();
                jobs.into_iter().map(|j|j.join().map_err(|_|"weight read thread panicked".to_string())?).collect::<std::result::Result<Vec<_>,String>>()
            })?;
            for (spec,content) in chunk.iter().zip(contents){
                let bytes=num(&spec["bytes"])?;
                let pointer=if defer_workspace && spec["role"]!="parameter" {
                    workspace_layout.push((buffers.len(),workspace_bytes));
                    workspace_bytes=workspace_bytes.checked_add(bytes.max(8)).and_then(|v|v.checked_add(255))
                        .ok_or("VAE workspace size overflow")?/256*256;0
                }else{arena.alloc(bytes.max(8))?};
                if num(&spec["id"])?!=buffers.len(){return Err("noncanonical allocation ids".into());}
                if let Some(data)=content {if data.len()>bytes{return Err("initializer exceeds buffer".into());}arena.upload(pointer,&data)?;}
                else if spec["role"]=="scratch"&&!spec["source"].is_null(){return Err("scratch buffer cannot have an initializer".into());}
                buffers.push(pointer);
            }
        }
        }
        let blas=arena.alloc(32<<20)?;let lt=arena.alloc(1<<20)?;let mut linear=ptr::null_mut();
        if unsafe{h3_linear_create(arena.stream,blas as _,32<<20,lt as _,1<<20,0,1,&mut linear)}!=0{return Err(unsafe{CStr::from_ptr(h3_linear_error())}.to_string_lossy().into_owned().into());}
        let mut conv=ptr::null_mut();if unsafe{h3_conv_create(&mut conv)}!=0{return Err(unsafe{CStr::from_ptr(h3_conv_error())}.to_string_lossy().into_owned().into());}
        let norm_scratch=arena.alloc(1024*1024)?;
        let mut kernels=HashMap::new();
        for (name,spec) in manifest["kernels"].as_object().ok_or("kernel table")?{
            let cubin=crate::startup_trace::read(root.join(text(&spec["file"])?))?;if hash(&cubin)!=text(&spec["sha256"])?{return Err("changed kernel CUBIN".into());}
            let mut module=ptr::null_mut();crate::startup_trace::measure("cuda_module","vae-module",cubin.len(),||check(unsafe{cuModuleLoadData(&mut module,cubin.as_ptr().cast())}))?;
            let mut function=ptr::null_mut();let symbol=CString::new(text(&spec["symbol"])?)?;
            crate::startup_trace::measure("cuda_function",symbol.to_string_lossy(),0,||check(unsafe{cuModuleGetFunction(&mut function,module,symbol.as_ptr())}))?;
            let shared=num(&spec["shared"])? as u32;
            if shared>0{check(unsafe{cuFuncSetAttribute(function,8,shared as i32)})?;}
            let mut count=0;loop{let(mut off,mut size)=(0,0);let rc=unsafe{cuFuncGetParamInfo(function,count,&mut off,&mut size)};if rc!=0{break;}if size!=4&&size!=8{return Err(format!("unsupported kernel param width {size}").into());}count+=1;}
            kernels.insert(name.clone(),Kernel{module,function,threads:num(&spec["warps"])? as u32*32,shared,params:count});
        }
        let hoist=if CastHoist::enabled(&manifest){
            let hoist=CastHoist::plan(&manifest)?;hoist.convert(&manifest,&mut arena,&buffers,&kernels)?;Some(std::sync::Arc::new(hoist))
        }else{None};
        eprintln!("loaded native VAE program: {} buffers, {} kernels, {} weight casts hoisted, {:.3}s",buffers.len(),kernels.len(),
            hoist.as_ref().map_or(0,|h|h.casts.len()),started.elapsed().as_secs_f64());
        let spatial=SpatialPlans::new(&std::sync::Arc::new(manifest),hoist)?;let manifest=spatial.top();
        Ok(Self{spatial,arena,linear,conv,norm_scratch,manifest,buffers,kernels,workspace_layout,workspace_bytes})
    }
    fn pointer(&self,spec:&Value)->Result<DevicePtr>{
        let index=num(&spec["buffer"])?;let offset=num(&spec["offset"])?;
        let size=num(&self.manifest["buffers"][index]["bytes"])?;
        if offset>size{return Err("tensor view offset out of allocation".into());}
        let pointer=*self.buffers.get(index).ok_or("invalid buffer index")?;
        if pointer==0{return Err("VAE workspace is not prepared".into());}Ok(pointer+offset as u64)
    }
    fn run(&self,verify_boundaries:bool)->Result<()> { self.run_nodes(verify_boundaries,true) }
    // Builds the convolution and attention plans of every node; launches nothing.
    fn prepare_metadata_plans(&self)->Result<()> {
        for op in arr(&self.manifest["ops"])? {
            if op["kind"]=="aten.convolution.default" {
                let args=arr(&op["args"])?;let(x,w)=(&args[0],&args[1]);let y=&op["outputs"][0];
                let mut spec=ConvSpec{ndim:5,dtype:0,allow_tf32:i32::from(self.manifest["backend"]["cudnn_allow_tf32"]==true),..Default::default()};
                spec.input_dims.copy_from_slice(&ints(&x["shape"])?);spec.input_strides.copy_from_slice(&ints(&x["stride"])?);
                spec.weight_dims.copy_from_slice(&ints(&w["shape"])?);spec.weight_strides.copy_from_slice(&ints(&w["stride"])?);
                spec.output_dims.copy_from_slice(&ints(&y["shape"])?);spec.output_strides.copy_from_slice(&ints(&y["stride"])?);
                spec.stride.copy_from_slice(&ints(&args[3])?);spec.padding.copy_from_slice(&ints(&args[4])?);spec.dilation.copy_from_slice(&ints(&args[5])?);
                let code=unsafe{h3_conv_prepare(self.conv,&spec,self.pointer(x)? as _,self.pointer(w)? as _,self.pointer(y)? as _,self.arena.stream)};
                if code!=0{return Err(unsafe{CStr::from_ptr(h3_conv_error())}.to_string_lossy().into_owned().into());}
            } else if op["kind"]=="cudnn_sdpa" {
                let args=arr(&op["args"])?;let(q,k,v)=(&args[0],&args[1],&args[2]);let out=&op["outputs"][0];
                let dims=ints(&q["shape"])?;let qs=ints(&q["stride"])?;let ks=ints(&k["stride"])?;let vs=ints(&v["stride"])?;let os=ints(&out["stride"])?;
                let code=unsafe{h3v_dense_prepare(dims.as_ptr(),qs.as_ptr(),ks.as_ptr(),vs.as_ptr(),os.as_ptr())};
                if code!=0{return Err(unsafe{CStr::from_ptr(h3v_dense_error())}.to_string_lossy().into_owned().into());}
            }
        }
        Ok(())
    }
    fn run_nodes(&self,verify_boundaries:bool,synchronize:bool)->Result<()> {
        if verify_boundaries && self.spatial.active!=(256,256){
            return Err("small tile requires an independent shape-matched output reference".into());
        }

        for (index,op) in arr(&self.manifest["ops"])?.iter().enumerate(){
            let outcome:Result<()>=match text(&op["kind"])?{
                "kernel"=>{
                    let kernel=self.kernels.get(text(&op["kernel"])?).ok_or("unknown CUBIN")?;
                    let mut values:Vec<u64>=arr(&op["args"])?.iter().map(|v|if v["type"]=="pointer"{self.pointer(v)}else{v["bits"].as_u64().ok_or_else(||"scalar bits".into())}).collect::<Result<_>>()?;
                    if values.len()!=kernel.params{return Err(format!("kernel {} ABI count {} != {}",index,values.len(),kernel.params).into());}
                    let symbol=text(&self.manifest["kernels"][text(&op["kernel"])?]["symbol"])?;
                    let kind=symbol.rsplit('_').next().and_then(|v|v.parse::<i32>().ok());
                    if self.spatial.active!=(256,256) && kind.is_some_and(|k|[1,4,5,6,7,8,11,13,14,15,18].contains(&k)) {
                        let (h,w)=self.spatial.active;
                        check(unsafe{h3v_generic_decoder_kernel(kind.unwrap(),values.as_ptr(),values.len() as _,7,(h/16) as _,(w/16) as _,self.arena.stream)})
                    }else{
                    let mut pointers:Vec<*mut c_void>=values.iter_mut().map(|v|(v as *mut u64).cast()).collect();
                    let grid=ints(&op["grid"])?;if grid.len()!=3{return Err("kernel grid rank".into());}
                    check(unsafe{cuLaunchKernel(kernel.function,grid[0] as _,grid[1] as _,grid[2] as _,kernel.threads,1,1,kernel.shared,self.arena.stream,pointers.as_mut_ptr(),ptr::null_mut())})
                    }
                },
                "mm"|"addmm"=>{
                    let args=arr(&op["args"])?;let bias=op["kind"]=="addmm";let base=usize::from(bias);
                    let x=&args[base];let w=&args[base+1];let out=&op["kwargs"]["out"];
                    let xs=ints(&x["shape"])?;let ws=ints(&w["shape"])?;let xst=ints(&x["stride"])?;let wst=ints(&w["stride"])?;
                    if xs.len()!=2||ws.len()!=2||xs[1]!=ws[0]||(!bias&&xst!=[xs[1],1]&&xst!=[1,xs[0]])||(bias&&xst!=[xs[1],1])||wst!=[1,ws[0]]{return Err(format!("unsupported compiled GEMM view at {index}: {op}").into());}
                    if x["dtype"]!="torch.float16"||w["dtype"]!="torch.float16"||out["dtype"]!="torch.float16"{return Err("decoder GEMM requires FP16".into());}
                    for k in ["alpha","beta"]{if !op["kwargs"][k].is_null() && op["kwargs"][k].as_f64()!=Some(1.0){return Err("unexpected GEMM coefficient".into());}}
                    let rc=unsafe{if bias{h3_linear_forward(self.linear,self.pointer(x)? as _,self.pointer(w)? as _,self.pointer(&args[0])? as _,self.pointer(out)? as _,xs[0] as _,ws[1] as _,xs[1] as _,2)}else{h3v_mm(self.linear,self.pointer(x)? as _,self.pointer(w)? as _,self.pointer(out)? as _,xs[0] as _,ws[1] as _,xs[1] as _,i32::from(xst==[1,xs[0]]))}};
                    if rc==0{Ok(())}else{Err(unsafe{CStr::from_ptr(h3_linear_error())}.to_string_lossy().into_owned().into())}
                },
                "cudnn_sdpa"=>{
                    let args=arr(&op["args"])?;let(q,k,v)=(&args[0],&args[1],&args[2]);let out=&op["outputs"][0];
                    let dims=ints(&q["shape"])?;let qs=ints(&q["stride"])?;let ks=ints(&k["stride"])?;let vs=ints(&v["stride"])?;let os=ints(&out["stride"])?;
                    if dims.len()!=4||q["shape"]!=k["shape"]||q["shape"]!=v["shape"]{return Err("invalid decoder SDPA dimensions".into());}
                    if q["dtype"]!="torch.float16"||k["dtype"]!="torch.float16"||v["dtype"]!="torch.float16"{return Err("decoder SDPA must be FP16".into());}
                    let scale=op["kwargs"]["scale"].as_f64().unwrap_or(1.0/(dims[3] as f64).sqrt()) as f32;
                    let rc=unsafe{h3v_dense(self.pointer(q)? as _,self.pointer(k)? as _,self.pointer(v)? as _,self.pointer(out)? as _,dims.as_ptr(),qs.as_ptr(),ks.as_ptr(),vs.as_ptr(),os.as_ptr(),scale,self.arena.stream)};
                    if rc==0{Ok(())}else{Err(unsafe{CStr::from_ptr(h3v_dense_error())}.to_string_lossy().into_owned().into())}
                },
                "aten.reflection_pad3d.default"|"aten.constant_pad_nd.default"=>{
                    let args=arr(&op["args"])?;let x=&args[0];let out=&op["outputs"][0];
                    let shape=ints(&x["shape"])?;let stride=ints(&x["stride"])?;let output=ints(&out["shape"])?;let pad=ints(&args[1])?;
                    if shape.len()!=5||output.len()!=5||pad.len()!=6||x["dtype"]!="torch.float32"{return Err("unsupported padding contract".into());}
                    if args.len()>2&&args[2].as_f64()!=Some(0.0){return Err("nonzero constant padding".into());}
                    check(unsafe{h3v_pad(self.pointer(x)? as _,self.pointer(out)? as _,shape.as_ptr(),stride.as_ptr(),output.as_ptr(),pad.as_ptr(),i32::from(op["kind"]=="aten.reflection_pad3d.default"),self.arena.stream)})
                },
                "aten.clone.default"=>{
                    let x=&op["args"][0];let out=&op["outputs"][0];
                    let shape=ints(&x["shape"])?;let stride=ints(&x["stride"])?;let output=ints(&out["shape"])?;
                    if shape.len()!=5 || shape!=output || x["dtype"]!="torch.float32" || out["dtype"]!="torch.float32" {
                        return Err("unsupported temporal encoder contiguous copy".into());
                    }
                    // Zero-padding is a pure strided copy, preserving the original
                    // frame-isolated GroupNorm's NCTHW <-> NTCHW materialization.
                    check(unsafe{h3v_pad(self.pointer(x)? as _,self.pointer(out)? as _,shape.as_ptr(),stride.as_ptr(),output.as_ptr(),[0i64;6].as_ptr(),0,self.arena.stream)})
                },
                "aten.convolution.default"=>{
                    let args=arr(&op["args"])?;let(x,w)=(&args[0],&args[1]);let y=&op["outputs"][0];
                    let mut spec=ConvSpec{ndim:5,dtype:0,allow_tf32:i32::from(self.manifest["backend"]["cudnn_allow_tf32"]==true),..Default::default()};
                    if x["dtype"]!="torch.float32"||w["dtype"]!="torch.float32"||args[6]!=false||args[8]!=1{return Err("unvalidated encoder convolution".into());}
                    spec.input_dims.copy_from_slice(&ints(&x["shape"])?);spec.input_strides.copy_from_slice(&ints(&x["stride"])?);
                    spec.weight_dims.copy_from_slice(&ints(&w["shape"])?);spec.weight_strides.copy_from_slice(&ints(&w["stride"])?);
                    spec.output_dims.copy_from_slice(&ints(&y["shape"])?);spec.output_strides.copy_from_slice(&ints(&y["stride"])?);
                    spec.stride.copy_from_slice(&ints(&args[3])?);spec.padding.copy_from_slice(&ints(&args[4])?);spec.dilation.copy_from_slice(&ints(&args[5])?);
                    let rc=unsafe{h3_conv_forward(self.conv,&spec,self.pointer(x)? as _,self.pointer(w)? as _,self.pointer(y)? as _,self.arena.stream)};
                    if rc!=0{return Err(format!("encoder conv {index}: {}",unsafe{CStr::from_ptr(h3_conv_error())}.to_string_lossy()).into());}
                    if !args[2].is_null(){let spatial=spec.output_dims[2..].iter().product::<i64>();let count=spec.output_dims.iter().product::<i64>();
                        check(unsafe{h3v_pointwise(self.pointer(y)? as _,self.pointer(&args[2])? as _,self.pointer(y)? as _,count,2,spatial,spec.output_dims[1],self.arena.stream)})?;}
                    Ok(())
                },
                "aten.native_group_norm.default"=>{
                    let args=arr(&op["args"])?;let output=arr(&op["outputs"])?;
                    let n=args[3].as_i64().ok_or("GN N")?;let c=args[4].as_i64().ok_or("GN C")?;let spatial=args[5].as_i64().ok_or("GN spatial")?;let groups=args[6].as_i64().ok_or("GN groups")?;
                    if n*c*8>1024*1024||spatial<=1||args[0]["dtype"]!="torch.float32"{return Err("unvalidated group norm shape/dtype".into());}
                    check(unsafe{h3v_group_norm(self.pointer(&args[0])? as _,self.pointer(&args[1])? as _,self.pointer(&args[2])? as _,self.pointer(&output[0])? as _,self.pointer(&output[1])? as _,self.pointer(&output[2])? as _,self.norm_scratch as _,n,c,spatial,groups,args[7].as_f64().ok_or("GN eps")? as f32,self.arena.stream)})
                },
                "aten.silu.default"|"aten.add.Tensor"=>{
                    let args=arr(&op["args"])?;let y=&op["outputs"][0];let add=op["kind"]=="aten.add.Tensor";
                    if args[0]["dtype"]!="torch.float32"||(!op["kwargs"]["alpha"].is_null()&&op["kwargs"]["alpha"]!=1){return Err("unvalidated encoder pointwise".into());}
                    if add&&args[0]["shape"]!=args[1]["shape"]{return Err("unexpected broadcast in residual add".into());}
                    let count=ints(&y["shape"])?.iter().product();
                    check(unsafe{h3v_pointwise(self.pointer(&args[0])? as _,if add{self.pointer(&args[1])? as _}else{ptr::null()},self.pointer(y)? as _,count,i32::from(add),0,0,self.arena.stream)})
                },
                kind=>return Err(format!("unsupported native plan op {kind}").into()),
            };
            outcome.map_err(|e|format!("op {index} {}: {e}",op["kind"]))?;
            if verify_boundaries {
                if let Some(checks)=op["checks"].as_array(){for (output_id,spec) in checks.iter().enumerate(){
                    let bytes=self.arena.download(self.pointer(spec)?,num(&spec["bytes"])?)?;let actual=hash(&bytes);let passed=actual==text(&spec["sha256"])?;
                    eprintln!("{}",json!({"op":index,"kind":op["kind"],"output":output_id,"passed":passed,"sha256":actual,"expected_sha256":spec["sha256"]}));
                    if !passed{return Err(format!("boundary mismatch op {index} {} output {output_id}",op["kind"]).into());}
                }}
            }
        }
        if synchronize {self.arena.sync()} else {Ok(())}
    }
    fn output(&self)->Result<Vec<u8>>{let spec=&self.manifest["outputs"][0];if num(&spec["offset"])?!=0{return Err("nonzero output offset".into());}self.arena.download(self.pointer(spec)?,num(&self.manifest["gold"]["bytes"])?) }
}
impl Drop for Program{fn drop(&mut self){let _=self.arena.sync();unsafe{h3v_dense_cleanup();h3_linear_destroy(self.linear);h3_conv_destroy(self.conv);}self.kernels.clear();}}
fn compare(actual:&[u8],expected:&[u8],dtype:&str)->Result<Value>{
    let width=if dtype=="torch.float16"{2}else if dtype=="torch.float32"{4}else{return Err("unvalidated comparison dtype".into())};
    if actual.len()!=expected.len()||actual.len()%width!=0{return Err("output byte size mismatch".into());}
    let mut mismatch=0;let mut nonfinite=0;let mut first=None;
    for(i,(a,b))in actual.chunks_exact(width).zip(expected.chunks_exact(width)).enumerate(){let (av,bv,mask)=if width==2{(u16::from_le_bytes([a[0],a[1]]) as u32,u16::from_le_bytes([b[0],b[1]]) as u32,0x7c00)}else{(u32::from_le_bytes(a.try_into()?),u32::from_le_bytes(b.try_into()?),0x7f800000)};if av&mask==mask||bv&mask==mask{nonfinite+=1;}if av!=bv{mismatch+=1;if first.is_none(){first=Some(i);}}}
    Ok(json!({"passed":mismatch==0&&nonfinite==0,"mismatch":mismatch,"nonfinite":nonfinite,"elements":actual.len()/width,"first_mismatch":first,"actual_sha256":hash(actual),"gold_sha256":hash(expected)}))
}
include!("tile_probe.rs");
fn work()->Result<()> {
    let args:Vec<_>=std::env::args().collect();

    if args.get(1).is_some_and(|x|x=="encode-video"){return full::encode_video(&args);}
    if args.get(1).is_some_and(|v|v=="tile-spatial"){return spatial_probe(&args);}
    if args.get(1).is_some_and(|v|v=="shape-plan"){return spatial_plan_probe(&args);}
if args.get(1).is_some_and(|x|x=="serve"){if args.len()!=3{return Err("usage: h3-video-vae serve SOCKET".into());}return full::serve(Path::new(&args[2]));}if args.get(1).is_some_and(|x|["encode","decode","decode-long","postquant"].contains(&x.as_str())){return full::work(&args);}
    if args.len()!=4{return Err("usage: h3-video-vae PROGRAM_ROOT OUTPUT_DIR REPEATS".into());}
    let root=Path::new(&args[1]);let out=Path::new(&args[2]);let repeats:usize=args[3].parse()?;if repeats<1{return Err("at least one replay".into());}fs::create_dir_all(out)?;
    let start=Instant::now();let program=Program::load(root)?;let load_seconds=start.elapsed().as_secs_f64();
    // Fixture validation may carry its own original-Python tile reference.
    // Full encode/decode never enters this branch or reads reference outputs.
    let gold_root=program.manifest.get("gold_root").unwrap_or(&program.manifest["audit_root"]);
    let gold=read_checked(Path::new(text(gold_root)?),&program.manifest["gold"])?;
    let mut results=Vec::new();for repeat in 0..repeats{let t=Instant::now();program.run(std::env::var("H3_VERIFY_BOUNDARIES").as_deref()==Ok("1"))?;let elapsed=t.elapsed().as_secs_f64();let output=program.output()?;let mut result=compare(&output,&gold,text(&program.manifest["gold"]["dtype"])?)?;result["repeat"]=json!(repeat);result["seconds"]=json!(elapsed);eprintln!("{}",result);let passed=result["passed"]==true;results.push(result);fs::write(out.join(format!("output{repeat}.raw")),&output)?;if !passed{break;}}
    let maps=fs::read_to_string("/proc/self/maps")?;fs::write(out.join("native.maps"),&maps)?;let clean=!["libtorch","libpython","libaten","libc10"].iter().any(|x|maps.to_lowercase().contains(x));
    let passed=clean&&results.len()==repeats&&results.iter().all(|r|r["passed"]==true);
    let report=json!({"passed":passed,"scope":program.manifest["stage"],"load_seconds":load_seconds,"wall_seconds":start.elapsed().as_secs_f64(),"program_sha256":hash(&crate::startup_trace::read(root.join("program.json"))?),"native_dependencies_clean":clean,"results":results});
    fs::write(out.join("report.json"),serde_json::to_string_pretty(&report)?+"\n")?;println!("{report}");if !passed{return Err("native decoder tile differs from original compiled reference".into());}Ok(())
}
fn main(){if let Err(e)=work(){eprintln!("{e}");std::process::exit(1);}}
