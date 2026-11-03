//! Literal Qwen3-VL conditioner forward. Static parameter bundle only in infer;
//! optional validation checkpoints observe results without supplying activations.
use super::*;
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

unsafe extern "C" {
    fn q3_cast_bf16(x:*const f32,out:*mut c_void,n:i64,stream:Stream)->c_int;
    fn q3_embedding(ids:*const i64,table:*const c_void,out:*mut c_void,rows:i64,width:c_int,stream:Stream)->c_int;
    fn q3_vision_rope(qkv:*const c_void,cos:*const f32,sin:*const f32,q:*mut c_void,k:*mut c_void,rows:i64,stream:Stream)->c_int;
    fn q3_text_rope(x:*const c_void,cos:*const c_void,sin:*const c_void,out:*mut c_void,rows:i64,heads:c_int,stream:Stream)->c_int;
    fn q3_vision_tables(pos:*const c_void,inv:*const f32,cos:*mut f32,sin:*mut f32,position:*mut c_void,grid_h:c_int,grid_w:c_int,stream:Stream)->c_int;
    fn q3_text_tables(position:*const i64,inv:*const f32,cos:*mut c_void,sin:*mut c_void,rows:c_int,stream:Stream)->c_int;
    fn q3_image_insert(hidden:*mut c_void,images:*const c_void,index:*const i64,tokens:c_int,add:c_int,stream:Stream)->c_int;
    fn h3_conv_create(context:*mut *mut c_void)->c_int;
    fn h3_conv_forward(context:*mut c_void,spec:*const ConvSpec,x:*const c_void,w:*const c_void,out:*mut c_void,stream:Stream)->c_int;
    fn h3_conv_error()->*const c_char;
    fn h3_conv_plan_tag(context:*mut c_void)->*const c_char;
    fn h3_conv_destroy(context:*mut c_void)->c_int;
}
#[repr(C)]
struct ConvSpec {
    ndim:i32,dtype:i32,transposed:i32,allow_tf32:i32,deterministic:i32,channels_last:i32,
    input_dims:[i64;5],input_strides:[i64;5],weight_dims:[i64;5],weight_strides:[i64;5],
    output_dims:[i64;5],output_strides:[i64;5],padding:[i64;3],stride:[i64;3],dilation:[i64;3],
}
impl Buffer {
    fn slice(self,offset:usize,bytes:usize)->Result<Self> {
        if offset>self.bytes || bytes>self.bytes-offset {return Err("device buffer view out of range".into());}
        Ok(Self{ptr:unsafe{self.ptr.cast::<u8>().add(offset).cast()},bytes})
    }
    fn prefix(self,bytes:usize)->Result<Self>{self.slice(0,bytes)}
}

struct Observer {root:PathBuf,manifest:Value,output:PathBuf,checks:Vec<Value>}
impl Observer {
    fn check(&mut self,arena:&Arena,name:&str,spec:&Value,buffer:Buffer)->Result<()> {
        if spec.get("file").is_none(){return Ok(());}
        let expected=raw(&self.root,spec)?;
        let actual=arena.read(buffer.prefix(expected.len())?)?;
        let comparison=if spec["dtype"]=="torch.bfloat16" {compare(&actual,&expected)?}
        else {json!({"passed":actual==expected,"bytes":actual.len(),"mismatching_bytes":actual.iter().zip(&expected).filter(|(a,b)|a!=b).count(),"sha256":hash(&actual)})};
        let pass=comparison["passed"]==true;
        let result=json!({"name":name,"expected_sha256":spec["sha256"],"comparison":comparison});
        eprintln!("checkpoint {}",serde_json::to_string(&result)?);
        self.checks.push(result);
        fs::write(self.output.join("checkpoints.json"),serde_json::to_vec_pretty(&self.checks)?)?;
        if !pass {
            fs::write(self.output.join("first-mismatch.raw"),actual)?;
            return Err(format!("conditioner first checkpoint mismatch: {name}").into());
        }
        Ok(())
    }
    fn module(&mut self,arena:&Arena,name:&str,buffer:Buffer)->Result<()> {
        let spec=self.manifest["module_fixtures"][name]["output"].clone();
        self.check(arena,name,&spec,buffer)
    }
    fn input(&mut self,arena:&Arena,name:&str,buffer:Buffer)->Result<()> {
        let fixture=&self.manifest["module_fixtures"][name];
        if fixture.is_null(){return Ok(());}
        let spec=input_spec(fixture)?.clone();
        self.check(arena,&format!("{name}.input"),&spec,buffer)
    }
}
#[derive(Clone, Copy)]
struct Layout {grid_h:usize,grid_w:usize,text_rows:usize,keyframe_count:usize}
impl Layout {
    /// No request shape: loading builds no attention plans.
    const WEIGHTS_ONLY:Layout=Layout{grid_h:0,grid_w:0,text_rows:0,keyframe_count:0};
    fn vision_sequence(self)->usize {self.grid_h*self.grid_w}
    fn vision_rows(self)->usize {self.keyframe_count*self.vision_sequence()}
    fn image_tokens(self)->usize {self.vision_rows()/4}
    fn sequence(self,kind:usize)->i64 {if kind==0{self.vision_sequence() as i64}else{self.text_rows as i64}}
}
struct Model {
    arena:Arena,params:BTreeMap<String,Buffer>,index:Value,conv:*mut c_void,
    parameter_end:usize,
    layout:Layout,attention_workspace:[Buffer;2],observer:Option<Observer>,conv_plan:String,observer_scope:String,
    /// Per-layer parameters of a parallel rank, resolved on its first forward.
    weights:Option<std::rc::Rc<Weights>>,
    /// Decoder projections of a sharded rank group (crate::text_shards).
    text:Option<TextShards>,
    startup_backend:Option<StartupPreparation>,
}
impl Drop for Model {
    fn drop(&mut self){self.startup_backend.take();let _=self.arena.sync();unsafe{h3_conv_destroy(self.conv);q3_attention_cleanup();}}
}
include!("startup_backend.rs");
impl Model {
    fn load(root:&Path,observer:Option<Observer>,layout:Layout,shards:Option<crate::text_shards::Layout>)->Result<Self> {
        let overlap=crate::weight_loader::flag("H3_QWEN_LOAD_INIT_OVERLAP")?;
        if overlap && shards.is_some() {return Err("Qwen initialization overlap requires the unsharded streaming loader".into());}
        if overlap && std::env::var("H3_QWEN_BACKEND_PREPARE").as_deref()==Ok("1") {
            // Tokenizer preparation owns no CUDA state. Its result gates model
            // readiness, but must not gate the first weight upload. Join it on
            // both success and failure so no preparation outlives this load.
            std::thread::scope(|scope|->Result<Self> {
                let tokenizer=std::thread::Builder::new().name("qwen-tokenizer-prepare".into())
                    .spawn_scoped(scope,move|| {
                        let _trace=crate::startup_trace::Span::new("qwen_tokenizer_prepare","overlap",0);
                        crate::prepare::cached_tokenizer(&root.join("tokenizer/tokenizer.json"))
                            .map(|_|()).map_err(|e|e.to_string())
                    })?;
                let loaded=Self::load_weights(root,observer,layout,shards,overlap);
                let prepared=crate::startup_trace::measure("qwen_tokenizer_wait","ready",0,||tokenizer.join())
                    .map_err(|_|"Qwen tokenizer preparation panicked")?;
                let model=loaded?;
                prepared?;
                Ok(model)
            })
        }else {
            Self::load_weights(root,observer,layout,shards,overlap)
        }
    }
    fn load_weights(root:&Path,observer:Option<Observer>,layout:Layout,shards:Option<crate::text_shards::Layout>,overlap:bool)->Result<Self> {
        use crate::text_shards::{sharded_weight,weight_name,LAYERS,PROJECTIONS};
        let prepare_cpu=||->Result<_> {
            let _trace=crate::startup_trace::Span::new("qwen_cpu_prepare",if overlap {"overlap"}else{"serial"},0);
            let manifest_trace=crate::startup_trace::Span::new("qwen_manifest_prepare","configs",0);
            let index:Value=serde_json::from_slice(&crate::startup_trace::read(root.join("model-manifest.json"))?)?;
            if index["schema"]!="h3-conditioner-static-model-v1" || index["selected_hidden_state_index"]!=50 ||
                index["activation_files_in_index"]!=0 || index["request_files_in_index"]!=0 {return Err("invalid static conditioner model index".into());}
            for (name,digest) in index["config_sha256"].as_object().ok_or("config hashes")? {
                if hash(&crate::startup_trace::read(root.join(name))?)!=digest.as_str().ok_or("config hash")? {return Err(format!("config changed: {name}").into());}
            }
            let mut tensors:Vec<(String,Value)>=["weights","buffers"].iter().flat_map(|group| index[group].as_object().unwrap().iter().map(|(n,s)|(n.clone(),s.clone()))).collect();
            if let Some(layout)=shards {
                for layer in 0..LAYERS {for (p,&(_,n,k)) in PROJECTIONS.iter().enumerate() {
                    let spec=&index["weights"][weight_name(layer,p)];
                    if spec.is_null() || shape(spec)?!=vec![n,k] || spec["dtype"]!="torch.bfloat16" {return Err(format!("sharded text weight {}",weight_name(layer,p)).into());}
                }}
                // A rank reads only the decoder projections it stores part of.
                tensors.retain(|(name,_)|sharded_weight(name).is_none_or(|(_,p)|layout.owned(p).is_some()));
            }
            let start=Instant::now();
            // Streaming upload owns complete tensors. Sharded ranks retain their
            // checked slice loader and its rank-local projection destinations.
            let config=crate::weight_loader::Config::from_env()?.filter(|_|shards.is_none());
            if crate::weight_loader::light::async_enabled()? && config.is_none(){
                return Err("background Qwen SHA requires the model streaming loader".into());
            }
            if overlap && config.is_none() {return Err("Qwen initialization overlap requires the unsharded streaming loader".into());}
            drop(manifest_trace);
            let pool=if let Some(config)=config {
                let jobs=tensors.iter().map(|(_,spec)| {
                    if spec.get("storage").is_some(){return Err("streaming conditioner requires direct weights".to_string());}
                    crate::weight_loader::Job::new(root,spec["file"].as_str().ok_or("tensor file")?,
                        usize::try_from(spec["bytes"].as_u64().ok_or("tensor bytes")?).map_err(|e|e.to_string())?,
                        Some(spec["sha256"].as_str().ok_or("tensor SHA256")?.to_owned()))
                }).collect::<std::result::Result<Vec<_>,String>>()?;
                Some(crate::startup_trace::measure("qwen_reader_start","streaming",0,||
                    crate::weight_loader::ReaderPool::start(jobs,config))?)
            }else{None};
            if !overlap && std::env::var("H3_QWEN_BACKEND_PREPARE").as_deref()==Ok("1") {
                // CPU preparation never owns a CUDA context or GPU pointer.
                let _trace=crate::startup_trace::Span::new("qwen_tokenizer_prepare","serial",0);
                crate::prepare::cached_tokenizer(&root.join("tokenizer/tokenizer.json"))?;
            }
            Ok((index,tensors,config,pool,start))
        };
        let ((index,tensors,config,mut pool,start),mut arena)=if overlap {
            // Keep CUDA initialization and every GPU operation on this owner.
            // Reader allocation and manifest checks run in a scoped CPU
            // worker, with the same bounded reader buffers.
            std::thread::scope(|scope|->Result<_> {
                let cpu=std::thread::Builder::new().name("qwen-load-cpu".into())
                    .spawn_scoped(scope,move||prepare_cpu().map_err(|e|e.to_string()))?;
                let arena=Arena::new();
                // Join on both success and failure. An unconsumed ReaderPool
                // drops by cancelling its channels and joining its readers.
                let prepared=cpu.join().map_err(|_|"Qwen CPU preparation panicked")?;
                Ok((prepared?,arena?))
            })?
        }else {
            (prepare_cpu()?,Arena::new()?)
        };
        eprintln!("conditioner load initialization overlap={overlap}");
        // The worker sees only validated shape metadata and its own handles.
        // Weight uploads remain on this owner and overlap vendor preparation.
        let startup_backend=if shards.is_none(){StartupPreparation::start()?}else{None};
        let mut params=BTreeMap::new();
        let text=shards.map(TextShards::new).transpose()?;
        let readers=match std::env::var("H3_MODEL_READ_PARALLELISM").as_deref() {
            Err(_)|Ok("8")=>8,Ok("32")=>32,_=>return Err("model read parallelism must be 8 or 32".into()),
        };
        let mut sharded=0;
        if let (Some(pool),Some(config))=(&mut pool,config) {
            let _trace=crate::startup_trace::Span::new("qwen_weight_upload","streaming",0);
            let mut upload=crate::startup_trace::measure("qwen_upload_setup","slots",0,||
                crate::weight_loader::device::Upload::new(arena.stream,config))?;
            let verify=std::env::var("H3_MODEL_LOAD_VERIFY_GPU").as_deref()==Ok("1");
            let sizes=tensors.iter().map(|(_,spec)|usize::try_from(spec["bytes"].as_u64().ok_or("tensor bytes missing")?)
                .map_err(|e|e.into())).collect::<Result<Vec<_>>>()?;
            let buffers=crate::startup_trace::measure("qwen_weight_allocation","model",0,||arena.alloc_model_tensors(&sizes))?;
            for ((name,_),&buffer) in tensors.iter().zip(&buffers) {
                params.insert(name.clone(),buffer);
            }
            unsafe{pool.consume_upload(&mut upload,|i,offset|buffers[i].ptr.cast::<u8>().add(offset).cast())}?;
            upload.finish()?;
            if verify {for (buffer,(_,spec)) in buffers.iter().zip(&tensors) {
                unsafe{upload.verify(buffer.ptr,buffer.bytes,spec["sha256"].as_str().unwrap())}?;
            }}
            eprintln!("conditioner streaming load: {} tensors, config={config:?}, GPU_SHA256_verified={verify}",params.len());
        }else {
            for group in tensors.chunks(readers) {
                let host:Result<Vec<_>>=std::thread::scope(|scope| {
                    let pending:Vec<_>=group.iter().map(|(name,spec)|scope.spawn(move||Ok((name.clone(),raw(root,spec)?)))).collect();
                    pending.into_iter().map(|p|p.join().map_err(|_|"model reader panicked")?).collect()
                });
                for (name,bytes) in host? {
                    match (text.as_ref(),shards,sharded_weight(&name)) {
                        (Some(text),Some(layout),Some((layer,p)))=>{
                            // The verified tensor's owned bytes go to their place in this rank's shard.
                            let (range,at)=layout.owned(p).ok_or("unowned text shard read")?;
                            let target=text.own(layer)?.slice(at,range.len())?;
                            cuda(unsafe{cudaMemcpyAsync(target.ptr,bytes[range].as_ptr().cast(),target.bytes,1,arena.stream)})?;
                            arena.sync()?;sharded+=1;
                        }
                        _=>{params.insert(name,arena.upload(&bytes)?);}
                    }
                }
            }
        }
        if let Some(layout)=shards {
            let owned=PROJECTIONS.iter().enumerate().filter(|(p,_)|layout.owned(*p).is_some()).count()*LAYERS;
            if sharded!=owned {return Err("incomplete text shard load".into());}
        }
        eprintln!("loaded {} tensors ({} text shard parts) with {} readers in {:.2}s",params.len(),sharded,readers,start.elapsed().as_secs_f64());
        let parameter_end=arena.allocations.len();
        let mut conv=ptr::null_mut();crate::startup_trace::measure("qwen_cuda_stage","conv-create",0,||native_error(unsafe{h3_conv_create(&mut conv)},h3_conv_error))?;
        let mut workspaces=vec![];
        for kind in 0..2 {
            // Weight-only preload has no request shape and creates no vendor
            // plans. The first real request admits and builds its actual keys.
            let mut size=0;if layout.sequence(kind as usize)>0 && (kind!=0 || layout.keyframe_count>0) {native_error(unsafe{q3_attention_workspace(kind,layout.sequence(kind as usize),&mut size)},q3_attention_error)?;}
            workspaces.push(arena.alloc(size)?);
        }
        Ok(Self{arena,params,index,conv,parameter_end,layout,attention_workspace:[workspaces[0],workspaces[1]],observer,conv_plan:String::new(),observer_scope:String::new(),weights:None,text,startup_backend})
    }
    fn parameter_boundary(&self)->Result<usize> {
        if self.parameter_end.checked_add(2)!=Some(self.arena.allocations.len()) {
            return Err("attention workspace allocation boundary changed".into());
        }
        let owned=self.arena.allocations[..self.parameter_end].iter().map(|b|(b.ptr as usize,b.bytes)).collect::<Vec<_>>();
        let views=self.params.values().map(|b|(b.ptr as usize,b.bytes)).collect::<Vec<_>>();
        crate::weight_loader::validate_tensor_allocations(&owned,&views)?;
        Ok(self.parameter_end)
    }
    fn p(&self,name:&str)->Result<Buffer>{self.params.get(name).copied().ok_or_else(||format!("missing parameter {name}").into())}
    /// A decoder layer's gathered projection views, after the compute stream
    /// waits for them; None when every rank keeps all weights.
    fn text_views(&mut self,layer:usize)->Result<Option<[Buffer;7]>> {
        let stream=self.arena.stream;
        match &mut self.text {Some(text)=>Ok(Some(text.layer(layer,stream)?)),None=>Ok(None)}
    }
    /// The layer's projection weights for the single-rank forward.
    fn text_weights(&mut self,layer:usize)->Result<[Buffer;7]> {
        if let Some(views)=self.text_views(layer)? {return Ok(views);}
        let mut weights=[Buffer{ptr:ptr::null_mut(),bytes:0};7];
        for (p,weight) in weights.iter_mut().enumerate() {*weight=self.p(&crate::text_shards::weight_name(layer,p))?;}
        Ok(weights)
    }
    /// Every read of the layer's views is queued on the compute stream.
    fn text_layer_done(&mut self,layer:usize)->Result<()> {
        let stream=self.arena.stream;
        match &mut self.text {Some(text)=>text.done(layer,stream),None=>Ok(())}
    }
    fn bf(&mut self,rows:usize,cols:usize)->Result<Buffer>{self.arena.alloc(rows*cols*2)}
    fn check(&mut self,name:&str,out:Buffer)->Result<()> {
        if let Some(observer)=&mut self.observer{observer.module(&self.arena,&format!("{}{name}",self.observer_scope),out)?;} Ok(())
    }
    fn input(&mut self,name:&str,out:Buffer)->Result<()> {
        if let Some(observer)=&mut self.observer{observer.input(&self.arena,&format!("{}{name}",self.observer_scope),out)?;} Ok(())
    }
    fn linear(&mut self,name:&str,x:Buffer,out:Buffer,rows:usize,k:usize,n:usize)->Result<()> {
        let weight=self.p(&format!("{name}.weight"))?;
        self.linear_weight(name,weight,x,out,rows,k,n)
    }
    fn linear_weight(&mut self,name:&str,weight:Buffer,x:Buffer,out:Buffer,rows:usize,k:usize,n:usize)->Result<()> {
        let spec=&self.index["weights"][format!("{name}.weight")];
        if shape(spec)?!=vec![n,k] || spec["dtype"]!="torch.bfloat16" {return Err(format!("linear spec {name}").into());}
        let bias=self.params.get(&format!("{name}.bias")).map_or(ptr::null_mut(),|b|b.ptr);
        self.input(name,x)?;
        native_error(unsafe{h3_linear_forward(self.arena.linear,x.ptr,weight.ptr,bias,out.ptr,rows as i32,n as i32,k as i32,0)},h3_linear_error)?;
        self.check(name,out)
    }
    fn norm(&mut self,name:&str,x:Buffer,out:Buffer,rows:usize,cols:usize,rms:bool)->Result<()> {
        self.input(name,x)?;
        let w=self.p(&format!("{name}.weight"))?;
        if rms {cuda(unsafe{q3_rms(x.ptr,w.ptr,out.ptr,rows as i64,cols as i32,1e-6,self.arena.stream)})?;}
        else {let b=self.p(&format!("{name}.bias"))?;cuda(unsafe{q3_layernorm(x.ptr,w.ptr,b.ptr,out.ptr,rows as i64,cols as i32,1e-6,self.arena.stream)})?;}
        self.check(name,out)
    }
    fn add(&self,x:Buffer,y:Buffer,out:Buffer,n:usize,ycols:usize)->Result<()> {
        cuda(unsafe{q3_binary(x.ptr,y.ptr,out.ptr,n as i64,0,ycols as i32,self.arena.stream)})
    }
    fn act(&mut self,name:&str,x:Buffer,out:Buffer,n:usize,kind:i32)->Result<()> {
        self.input(name,x)?;cuda(unsafe{q3_activation(x.ptr,out.ptr,n as i64,kind,self.arena.stream)})?;self.check(name,out)
    }
    fn attention(&self,kind:usize,q:Buffer,k:Buffer,v:Buffer,out:Buffer)->Result<()> {
        let ws=self.attention_workspace[kind];
        let scale=if kind==0 {72f64.powf(-0.5) as f32}else{128f64.powf(-0.5) as f32};
        native_error(unsafe{q3_attention_forward(kind as i32,self.layout.sequence(kind),q.ptr,k.ptr,v.ptr,out.ptr,ws.ptr,ws.bytes,scale,self.arena.stream)},q3_attention_error)
    }
    fn merger(&mut self,name:&str,x:Buffer,norm:Buffer,fc:Buffer,out:Buffer,post:bool,rows:usize)->Result<()> {
        let merged=rows/4;
        self.input(name,x)?;
        self.norm(&format!("{name}.norm"),x,norm,if post{merged}else{rows},if post{4608}else{1152},false)?;
        self.linear(&format!("{name}.linear_fc1"),norm,fc,merged,4608,4608)?;
        self.act(&format!("{name}.act_fn"),fc,fc,merged*4608,2)?;
        self.linear(&format!("{name}.linear_fc2"),fc,out,merged,4608,5120)?;
        self.check(name,out)
    }
    fn vision(&mut self,prepared:&Path)->Result<(Buffer,[Buffer;3])> {
        self.vision_grids(prepared,&vec![(self.layout.grid_h,self.layout.grid_w);self.layout.keyframe_count])
    }
    fn vision_grids(&mut self,prepared:&Path,grids:&[(usize,usize)])->Result<(Buffer,[Buffer;3])> {
        self.vision_grids_file(prepared,grids,"pixel_values.raw")
    }
    fn vision_grids_file(&mut self,prepared:&Path,grids:&[(usize,usize)],file:&str)->Result<(Buffer,[Buffer;3])> {
        let rows=grids.iter().map(|&(h,w)|h*w).sum::<usize>();let merged_rows=rows/4;
        let max_sequence=grids.iter().map(|&(h,w)|h*w).max().ok_or("missing vision grids")?;
        let mut needed=0;
        for &(h,w) in grids {let mut size=0;native_error(unsafe{q3_attention_workspace(0,(h*w) as i64,&mut size)},q3_attention_error)?;needed=needed.max(size);}
        if self.attention_workspace[0].bytes<needed {self.attention_workspace[0]=self.arena.alloc(needed)?;}
        let pixels=self.arena.upload(&crate::startup_trace::read(prepared.join(file))?)?;
        if pixels.bytes!=rows*1536*4 {return Err("pixel tensor size".into());}
        let input=self.bf(rows,1536)?;
        cuda(unsafe{q3_cast_bf16(pixels.ptr.cast(),input.ptr,(rows*1536) as i64,self.arena.stream)})?;
        self.input("visual.patch_embed",input)?;
        let hidden=self.bf(rows,1152)?;let norm=self.bf(rows,1152)?;let residual=self.bf(rows,1152)?;
        let qkv=self.bf(rows,3456)?;let q=self.bf(rows,1152)?;let k=self.bf(rows,1152)?;
        let attn=self.bf(rows,1152)?;let mlp=self.bf(rows,4304)?;
        // Preserve the verified two-grid table kernel, copying its first grid
        // for each reference. Variable geometries retain original batch order.
        let cos=self.arena.alloc(rows*72*4)?;let sin=self.arena.alloc(rows*72*4)?;
        let position=self.bf(rows,1152)?;
        let temp_cos=self.arena.alloc(2*max_sequence*72*4)?;
        let temp_sin=self.arena.alloc(2*max_sequence*72*4)?;
        let temp_position=self.bf(2*max_sequence,1152)?;
        let spec=ConvSpec{ndim:5,dtype:2,transposed:0,allow_tf32:1,deterministic:0,channels_last:0,
            input_dims:[rows as i64,3,2,16,16],input_strides:[1536,512,256,16,1],
            weight_dims:[1152,3,2,16,16],weight_strides:[1536,512,256,16,1],
            output_dims:[rows as i64,1152,1,1,1],output_strides:[1152,1,1,1,1],padding:[0;3],stride:[2,16,16],dilation:[1;3]};
        let w=self.p("visual.patch_embed.proj.weight")?;let b=self.p("visual.patch_embed.proj.bias")?;
        native_error(unsafe{h3_conv_forward(self.conv,&spec,input.ptr,w.ptr,hidden.ptr,self.arena.stream)},h3_conv_error)?;
        self.conv_plan=unsafe{CStr::from_ptr(h3_conv_plan_tag(self.conv))}.to_string_lossy().into_owned();
        self.add(hidden,b,hidden,rows*1152,1152)?;self.check("visual.patch_embed",hidden)?;
        let table=self.p("visual.pos_embed.weight")?;let inv=self.p("visual.rotary_pos_emb.inv_freq")?;
        let mut offset=0;
        for &(h,w) in grids {
            let n=h*w;
            cuda(unsafe{q3_vision_tables(table.ptr,inv.ptr.cast(),temp_cos.ptr.cast(),temp_sin.ptr.cast(),temp_position.ptr,h as i32,w as i32,self.arena.stream)})?;
            for (source,destination,stride) in [(temp_cos,cos,72*4),(temp_sin,sin,72*4),(temp_position,position,1152*2)] {
                let dest=destination.slice(offset*stride,n*stride)?;
                cuda(unsafe{cudaMemcpyAsync(dest.ptr,source.ptr,n*stride,3,self.arena.stream)})?;
            }
            offset+=n;
        }
        self.add(hidden,position,hidden,rows*1152,0)?;
        if let Some(observer)=&mut self.observer {
            for (i,buffer) in [cos,sin].into_iter().enumerate() {
                let expected=observer.manifest["module_fixtures"][format!("{}visual.blocks.0",self.observer_scope)]["kwargs"]["position_embeddings"][i].clone();
                observer.check(&self.arena,&format!("{}vision_rotary.{i}",self.observer_scope),&expected,buffer)?;
            }
        }
        let deep=[self.bf(merged_rows,5120)?,self.bf(merged_rows,5120)?,self.bf(merged_rows,5120)?];
        let merged=self.bf(merged_rows,5120)?;let mergefc=self.bf(merged_rows,4608)?;
        for layer in 0..27 {
            let name=format!("visual.blocks.{layer}");self.input(&name,hidden)?;
            self.norm(&format!("{name}.norm1"),hidden,norm,rows,1152,false)?;
            self.linear(&format!("{name}.attn.qkv"),norm,qkv,rows,1152,3456)?;
            cuda(unsafe{q3_vision_rope(qkv.ptr,cos.ptr.cast(),sin.ptr.cast(),q.ptr,k.ptr,rows as i64,self.arena.stream)})?;
            let mut offset=0;
            for &(h,w) in grids {
                let sequence=h*w;let ws=self.attention_workspace[0];
                let qi=q.slice(offset*1152*2,sequence*1152*2)?;let ki=k.slice(offset*1152*2,sequence*1152*2)?;
                let vi=qkv.slice((offset*3456+2304)*2,((sequence-1)*3456+1152)*2)?;
                let oi=attn.slice(offset*1152*2,sequence*1152*2)?;
                native_error(unsafe{q3_attention_forward(0,sequence as i64,qi.ptr,ki.ptr,vi.ptr,oi.ptr,ws.ptr,ws.bytes,72f64.powf(-0.5) as f32,self.arena.stream)},q3_attention_error)?;
                offset+=sequence;
            }
            self.linear(&format!("{name}.attn.proj"),attn,residual,rows,1152,1152)?;
            self.add(hidden,residual,hidden,rows*1152,0)?;
            self.norm(&format!("{name}.norm2"),hidden,norm,rows,1152,false)?;
            self.linear(&format!("{name}.mlp.linear_fc1"),norm,mlp,rows,1152,4304)?;
            self.act(&format!("{name}.mlp.act_fn"),mlp,mlp,rows*4304,1)?;
            self.linear(&format!("{name}.mlp.linear_fc2"),mlp,residual,rows,4304,1152)?;
            self.add(hidden,residual,hidden,rows*1152,0)?;self.check(&name,hidden)?;
            if let Some(index)=[8,16,24].iter().position(|&n|n==layer) {
                self.merger(&format!("visual.deepstack_merger_list.{index}"),hidden,norm,mergefc,deep[index],true,rows)?;
            }
            if crate::debug_audit::enabled(){eprintln!("vision layer {layer} complete");}
        }
        self.merger("visual.merger",hidden,norm,mergefc,merged,false,rows)?;
        Ok((merged,deep))
    }
    fn text(&mut self,prepared:&Path,vision:Option<(Buffer,[Buffer;3])>)->Result<Buffer> {
        self.text_multimodal(prepared,vision,None)
    }
    fn text_multimodal(&mut self,prepared:&Path,vision:Option<(Buffer,[Buffer;3])>,video:Option<(Buffer,[Buffer;3])>)->Result<Buffer> {
        let s=self.layout.text_rows;
        let idbytes=crate::startup_trace::read(prepared.join("input_ids.raw"))?;
        if idbytes.len()!=s*8 {return Err("unsupported native conditioner text length".into());}
        let ids=self.arena.upload(&idbytes)?;
        let config:Value=serde_json::from_slice(&crate::startup_trace::read(prepared.join("../model-config.json"))?)?;
        let mut insertions=vec![];
        for (features,key) in [(vision,"image_token_id"),(video,"video_token_id")] {
            let id=config[key].as_i64().ok_or("modality token id")?;
            let indices:Vec<i64>=idbytes.chunks_exact(8).enumerate().filter_map(|(i,b)|(i64::from_le_bytes(b.try_into().unwrap())==id).then_some(i as i64)).collect();
            let count=features.map_or(0,|(v,_)|v.bytes/(5120*2));
            if indices.len()!=count {return Err("vision token count".into());}
            if let Some((features,deep))=features {
                let index=self.arena.upload(&indices.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
                insertions.push((features,deep,index,count));
            }
        }
        let hidden=self.bf(s,5120)?;let norm=self.bf(s,5120)?;let residual=self.bf(s,5120)?;
        let q=self.bf(s,8192)?;let qnorm=self.bf(s,8192)?;
        let k=self.bf(s,1024)?;let knorm=self.bf(s,1024)?;let v=self.bf(s,1024)?;
        let attention=self.bf(s,8192)?;let gate=self.bf(s,25600)?;let up=self.bf(s,25600)?;
        let table=self.p("language_model.embed_tokens.weight")?;
        cuda(unsafe{q3_embedding(ids.ptr.cast(),table.ptr,hidden.ptr,s as i64,5120,self.arena.stream)})?;
        self.check("language_model.embed_tokens",hidden)?;
        for &(features,_,indices,count) in &insertions {cuda(unsafe{q3_image_insert(hidden.ptr,features.ptr,indices.ptr.cast(),count as i32,0,self.arena.stream)})?;}
        let positions=self.arena.upload(&crate::startup_trace::read(prepared.join("position_ids.raw"))?)?;
        let inv=self.p("language_model.rotary_emb.inv_freq")?;let cos=self.bf(s,128)?;let sin=self.bf(s,128)?;
        cuda(unsafe{q3_text_tables(positions.ptr.cast(),inv.ptr.cast(),cos.ptr,sin.ptr,s as i32,self.arena.stream)})?;
        if let Some(observer)=&mut self.observer {
            for (i,buffer) in [cos,sin].into_iter().enumerate() {
                let expected=observer.manifest["module_fixtures"]["language_model.rotary_emb"]["output"][i].clone();
                observer.check(&self.arena,&format!("text_rotary.{i}"),&expected,buffer)?;
            }
        }
        for layer in 0..50 {
            let name=format!("language_model.layers.{layer}");self.input(&name,hidden)?;
            let w=self.text_weights(layer)?;
            self.norm(&format!("{name}.input_layernorm"),hidden,norm,s,5120,true)?;
            self.linear_weight(&format!("{name}.self_attn.q_proj"),w[0],norm,q,s,5120,8192)?;
            self.norm(&format!("{name}.self_attn.q_norm"),q,qnorm,s*64,128,true)?;
            self.linear_weight(&format!("{name}.self_attn.k_proj"),w[1],norm,k,s,5120,1024)?;
            self.norm(&format!("{name}.self_attn.k_norm"),k,knorm,s*8,128,true)?;
            self.linear_weight(&format!("{name}.self_attn.v_proj"),w[2],norm,v,s,5120,1024)?;
            cuda(unsafe{q3_text_rope(qnorm.ptr,cos.ptr,sin.ptr,q.ptr,s as i64,64,self.arena.stream)})?;
            cuda(unsafe{q3_text_rope(knorm.ptr,cos.ptr,sin.ptr,k.ptr,s as i64,8,self.arena.stream)})?;
            self.attention(1,q,k,v,attention)?;
            self.linear_weight(&format!("{name}.self_attn.o_proj"),w[3],attention,residual,s,8192,5120)?;
            self.add(hidden,residual,hidden,s*5120,0)?;
            self.norm(&format!("{name}.post_attention_layernorm"),hidden,norm,s,5120,true)?;
            self.linear_weight(&format!("{name}.mlp.gate_proj"),w[4],norm,gate,s,5120,25600)?;
            self.act(&format!("{name}.mlp.act_fn"),gate,gate,s*25600,0)?;
            self.linear_weight(&format!("{name}.mlp.up_proj"),w[5],norm,up,s,5120,25600)?;
            cuda(unsafe{q3_binary(gate.ptr,up.ptr,gate.ptr,(s*25600) as i64,1,0,self.arena.stream)})?;
            self.linear_weight(&format!("{name}.mlp.down_proj"),w[6],gate,residual,s,25600,5120)?;
            self.text_layer_done(layer)?;
            self.add(hidden,residual,hidden,s*5120,0)?;self.check(&name,hidden)?;
            if layer<3 {for &(_,deep,indices,count) in &insertions {cuda(unsafe{q3_image_insert(hidden.ptr,deep[layer].ptr,indices.ptr.cast(),count as i32,1,self.arena.stream)})?;}}
            if crate::debug_audit::enabled(){eprintln!("text layer {layer} complete");}
        }
        Ok(hidden)
    }
}

pub fn infer_ref2va(model_root:&Path,prompt:&Path,descriptor:&Path,output:&Path,gold_root:Option<&Path>)->Result<Value> {
    if !output.is_absolute() || output.exists(){return Err("output must be a fresh absolute directory".into());}
    let (_,manifest_sha)=crate::dynamic_request::checked_model(model_root)?;
    let start=Instant::now();let prepared=output.join("prepared");
    let prep=crate::ref2va::run(model_root,prompt,descriptor,&prepared)?;
    let grids:Vec<(usize,usize)>=serde_json::from_value(prep["grids"].clone())?;
    let video_grids:Vec<(usize,usize)>=match prep.get("video_grids"){Some(v)=>serde_json::from_value(v.clone())?,None=>vec![]};
    let &(h,w)=grids.first().or_else(||video_grids.first()).ok_or("vision grids")?;
    let text_rows=prep["token_count"].as_u64().ok_or("text rows")? as usize;
    fs::copy(model_root.join("model_config/config.json"),output.join("model-config.json"))?;
    audit(output,"start")?;
    let observer=if let Some(root)=gold_root {Some(Observer{root:root.to_path_buf(),manifest:serde_json::from_slice(&crate::startup_trace::read(root.join("manifest.json"))?)?,output:output.to_path_buf(),checks:vec![]})}else{None};
    let mut model=Model::load(model_root,observer,Layout{grid_h:h,grid_w:w,text_rows,keyframe_count:1},None)?;
    let vision=if grids.is_empty(){None}else{Some(model.vision_grids(&prepared,&grids)?)};
    // Original Qwen runs one vision forward per modality, retaining GEMM batch shapes.
    model.observer_scope=if grids.is_empty(){String::new()}else{"video.".into()};
    let video=if video_grids.is_empty(){None}else{Some(model.vision_grids_file(&prepared,&video_grids,"pixel_values_videos.raw")?)};
    model.observer_scope.clear();
    let hidden=model.text_multimodal(&prepared,vision,video)?;
    let conditioning=model.arena.read(hidden)?;
    let nonfinite=conditioning.chunks_exact(2).filter(|b|!f32::from_bits((u16::from_le_bytes([b[0],b[1]]) as u32)<<16).is_finite()).count();
    if nonfinite!=0{return Err("nonfinite Ref2VA conditioning".into());}
    fs::write(output.join("conditioning.raw"),&conditioning)?;
    fs::copy(prepared.join("text_token_tags.raw"),output.join("text_token_tags.raw"))?;
    audit(output,"complete")?;
    let report=json!({"passed":true,"scope":"native Qwen from normalized image/video RGB and audio labels; normalization external",
        "model_manifest_sha256":manifest_sha,"gold_activations_read_for_computation":false,"validation_mode":gold_root.is_some(),
        "conditioning":{"sha256":hash(&conditioning),"shape":[1,text_rows,5120],"bytes":conditioning.len(),"nonfinite":nonfinite},
        "text_token_tags_sha256":hash(&crate::startup_trace::read(output.join("text_token_tags.raw"))?),"preparation":prep,
        "checkpoint_count":model.observer.as_ref().map_or(0,|o|o.checks.len()),"wall_seconds":start.elapsed().as_secs_f64()});
    fs::write(output.join("report.json"),serde_json::to_vec_pretty(&report)?)?;Ok(report)
}

pub fn infer_with_size(model_root:&Path,prompt:&Path,canvases:&[&Path],output:&Path,gold_root:Option<&Path>,width:usize,height:usize)->Result<Value> {
    fs::create_dir_all(output)?;audit(output,"start")?;
    let start=Instant::now();
    let prepared=output.join("prepared");
    let prep=crate::prepare::run_with_size(model_root,prompt,canvases,&prepared,width,height)?;
    let text_rows=prep["token_count"].as_u64().ok_or("prepared token count")? as usize;
    if text_rows==0 || text_rows>i32::MAX as usize {return Err("unsupported text sequence size".into());}
    let layout=Layout{grid_h:prep["grid_height"].as_u64().ok_or("grid height")? as usize,grid_w:prep["grid_width"].as_u64().ok_or("grid width")? as usize,text_rows,keyframe_count:prep["keyframe_count"].as_u64().ok_or("keyframe count")? as usize};
    fs::copy(model_root.join("model_config/config.json"),output.join("model-config.json"))?;
    let observer=if let Some(root)=gold_root {Some(Observer{root:root.to_path_buf(),manifest:serde_json::from_slice(&crate::startup_trace::read(root.join("manifest.json"))?)?,output:output.to_path_buf(),checks:vec![]})}else{None};
    let mut model=Model::load(model_root,observer,layout,None)?;
    let vision=if layout.keyframe_count>0 {Some(model.vision(&prepared)?)}else{None};
    let hidden=model.text(&prepared,vision)?;
    let conditioning=model.arena.read(hidden)?;
    let nonfinite=conditioning.chunks_exact(2).filter(|b|!f32::from_bits((u16::from_le_bytes([b[0],b[1]]) as u32)<<16).is_finite()).count();
    if nonfinite>0 {return Err("nonfinite native conditioning".into());}
    fs::write(output.join("conditioning.raw"),&conditioning)?;
    fs::copy(prepared.join("text_token_tags.raw"),output.join("text_token_tags.raw"))?;
    let mut plans=vec![];
    for kind in 0..2 {if kind==0 && layout.keyframe_count==0 {continue;}plans.push(unsafe{CStr::from_ptr(q3_attention_plan_name(kind,layout.sequence(kind as usize)))}.to_string_lossy().into_owned());}
    audit(output,"complete")?;
    let report=json!({"schema":"h3-native-conditioner-v1","passed":true,"native_process_clean":true,
        "gold_activations_read_for_computation":false,"validation_mode":gold_root.is_some(),
        "model_manifest_sha256":hash(&crate::startup_trace::read(model_root.join("model-manifest.json"))?),
        "source_export_manifest_sha256":model.index["source_manifest_sha256"],"selected_hidden_state_index":50,
        "conditioning":{"file":"conditioning.raw","dtype":"bfloat16","shape":[1,text_rows,5120],"bytes":conditioning.len(),"sha256":hash(&conditioning),"nonfinite":nonfinite},
        "text_token_tags":{"file":"text_token_tags.raw","dtype":"int64","shape":[text_rows],"sha256":hash(&crate::startup_trace::read(output.join("text_token_tags.raw"))?)},
        "preparation":prep,"conv_plan":model.conv_plan,"attention_plans":plans,
        "checkpoint_count":model.observer.as_ref().map_or(0,|v|v.checks.len()),"wall_seconds":start.elapsed().as_secs_f64()});
    fs::write(output.join("report.json"),serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

// Appended to the verified original native/model.rs. Existing forward methods
// and the accepted CUDA/cuDNN/cuBLAS shared object remain unchanged.
unsafe extern "C" {fn cudaMemGetInfo(free:*mut usize,total:*mut usize)->c_int;}

fn server_memory(model:&Model)->Result<Value> {
    model.arena.sync()?;
    let (mut free,mut total)=(0usize,0usize);
    cuda(unsafe{cudaMemGetInfo(&mut free,&mut total)})?;
    Ok(json!({"device_total_bytes":total,"device_free_bytes":free,"device_used_bytes":total-free,
        "arena_allocated_bytes":model.arena.allocations.iter().map(|b|b.bytes).sum::<usize>(),
        "arena_allocation_count":model.arena.allocations.len(),
        "scope":"device used includes all concurrently resident processes; arena counts this conditioner only"}))
}

fn server_identity(model:&Model)->Result<String> {
    let params:Vec<_>=model.params.iter().map(|(name,b)|
        json!({"name":name,"pointer":format!("{:p}",b.ptr),"bytes":b.bytes})).collect();
    Ok(hash(&serde_json::to_vec(&json!({"parameters":params,"text_shards":model.text.as_ref().map(|t|t.identity()),
        "stream":format!("{:p}",model.arena.stream),
        "linear":format!("{:p}",model.arena.linear),"conv":format!("{:p}",model.conv)}))?))
}

// Generic request state, appended after all unchanged accepted forward methods.
unsafe extern "C" {fn q3_attention_cache_size()->usize;fn q3_conv_cache_size(context:*mut c_void)->usize;fn q3_conv_cache_clear(context:*mut c_void)->c_int;fn h3_linear_fallback_reset();fn h3_linear_fallback_counts(out:*mut u64,n:c_int)->c_int;}
use std::sync::atomic::{AtomicU64,Ordering};
pub(super) static DYNAMIC_ALLOCATIONS:AtomicU64=AtomicU64::new(0);

fn release_after(model:&mut Model,keep:usize)->Result<(usize,usize)> {
    model.arena.sync()?;
    if keep>model.arena.allocations.len(){return Err("persistent allocation boundary changed".into());}
    let buffers=model.arena.allocations.split_off(keep);
    let bytes=buffers.iter().map(|b|b.bytes).sum();let count=buffers.len();
    for buffer in buffers {cuda(unsafe{cudaFree(buffer.ptr)})?;}
    Ok((count,bytes))
}

/// Returns (changed, text_pending). With `defer_text`, the text attention plan and
/// workspace are left to attach_text_workspace, called once the vision tower is
/// queued so the host builds the plan while the GPU runs vision.
fn reconfigure(model:&mut Model,layout:Layout,weights_end:usize,clear:bool,defer_text:bool,parallel:Option<usize>)->Result<(bool,bool)> {
    let changed=(model.layout.grid_h,model.layout.grid_w,model.layout.text_rows,model.layout.keyframe_count)!=(layout.grid_h,layout.grid_w,layout.text_rows,layout.keyframe_count);
    if !changed && !clear{return Ok((false,false));}
    // No work can refer to old workspaces/plans after this synchronization.
    release_after(model,weights_end)?;
    if clear{
        native_error(unsafe{q3_attention_cleanup()},q3_attention_error)?;
        native_error(unsafe{q3_cp_attention_cleanup()},q3_cp_attention_error)?;
        native_error(unsafe{q3_conv_cache_clear(model.conv)},h3_conv_error)?;
    }
    let mut buffers=vec![];
    for kind in 0..if defer_text{1}else{2} {
        let mut bytes=0;
        if kind!=0 || layout.keyframe_count>0 {match parallel {
            // A rank group plans its own head-group attention instead.
            Some(world)=>native_error(unsafe{q3_cp_attention_workspace(kind,world as c_int,layout.sequence(kind as usize),&mut bytes)},q3_cp_attention_error)?,
            None=>native_error(unsafe{q3_attention_workspace(kind,layout.sequence(kind as usize),&mut bytes)},q3_attention_error)?,
        }}
        buffers.push(model.arena.alloc(bytes)?);
    }
    model.layout=layout;model.attention_workspace=[buffers[0],if defer_text{model.attention_workspace[1]}else{buffers[1]}];
    Ok((true,defer_text))
}
/// Build the text attention plan and allocate its workspace into the persistent
/// slot after the vision workspace, ahead of the request's temporary buffers.
fn attach_text_workspace(model:&mut Model,weights_end:usize)->Result<()> {
    let mut bytes=0;
    native_error(unsafe{q3_attention_workspace(1,model.layout.sequence(1),&mut bytes)},q3_attention_error)?;
    let buffer=model.arena.alloc(bytes)?;
    let last=model.arena.allocations.pop().ok_or("text workspace allocation missing")?;
    if weights_end+1>model.arena.allocations.len(){return Err("persistent allocation boundary changed".into());}
    model.arena.allocations.insert(weights_end+1,last);
    model.attention_workspace[1]=buffer;
    Ok(())
}

struct ConditionerServer {
    group:Option<Group>,
    model:Option<Model>,fixed:Option<Value>,identity:String,requests:usize,rejected:usize,
    initial_load:f64,weights_end:usize,plan_generation:usize,in_gpu:bool,cache:crate::plan_cache::Cache,
    /// This rank's part of a sharded group's decoder projections.
    text_shards:Option<crate::text_shards::Layout>,
    host_weights:Option<HostWeights>,offloaded:bool,gpu_restore_count:usize,
}
impl ConditionerServer {
    /// Device bytes a first load adds: every unsharded tensor plus, on a sharded
    /// rank, its shards and gather slots.
    fn initial_weight_bytes(&self,index:&Value)->Result<usize> {
        let mut bytes=33usize<<20;
        for spec in index["weights"].as_object().ok_or("weights")?.iter().chain(index["buffers"].as_object().ok_or("buffers")?.iter())
            .filter(|(name,_)|self.text_shards.is_none() || crate::text_shards::sharded_weight(name).is_none()).map(|(_,s)|s) {
            bytes=bytes.checked_add(spec["bytes"].as_u64().unwrap_or(u64::MAX) as usize).ok_or("weight bytes overflow")?;
        }
        bytes.checked_add(self.text_shards.map_or(0,|l|l.resident_bytes())).ok_or_else(||"weight bytes overflow".into())
    }
    /// Join the group's stream and, on a sharded rank, map every rank's shards
    /// (a collective every rank reaches during the same request).
    fn attach(&mut self,model:&mut Model)->Result<()> {
        if let Some(group)=&mut self.group{group.bind(model.arena.stream)?;}
        if let (Some(group),Some(text))=(self.group.as_ref(),model.text.as_mut()){text.map(group)?;}
        Ok(())
    }
    /// Rank-consistent memory admission: any rank's shortfall rejects on all.
    fn admit_memory(&self,admitted:bool,error:&str)->Result<()> {
        let rejected=match &self.group {Some(group)=>group.any(!admitted)?,None=>!admitted};
        if rejected{return Err(error.into());}
        Ok(())
    }
}
impl ConditionerServer {
    fn preload(&mut self,args:&[String])->Result<Value> {
        if args.len()!=2 || self.model.is_some() {return Err("preload requires one unloaded conditioner model".into());}
        let root=fs::canonicalize(&args[1])?;
        let (_,sha)=crate::dynamic_request::checked_model(&root)?;
        self.in_gpu=true;let started=Instant::now();
        let mut model=Model::load(&root,None,Layout{grid_h:0,grid_w:0,text_rows:0,keyframe_count:0},self.text_shards)?;
        model.arena.sync()?;
        if unsafe{q3_attention_cache_size()}!=0 || unsafe{q3_conv_cache_size(model.conv)}!=0 {
            return Err("weight preload unexpectedly created request plans".into());
        }
        // The weight-only invariant above is unchanged. Preparing vendor plans
        // afterward reads no activations and executes no forward. Missing exact
        // serialized keys always retain the original builder.
        let mut prepared_attention_plans=0;
        if self.text_shards.is_none() {
            let metadata=std::env::var_os("H3_QWEN_PRELOAD_PLAN_FILE").map(PathBuf::from);
            let shapes=match metadata {
                Some(path) if path.is_file()=>Some(fs::read_to_string(path)?),
                Some(_)=>None, // The verified bundle is optional until published atomically.
                None=>std::env::var("H3_QWEN_PRELOAD_PLANS").ok(),
            };
            if let Some(raw)=shapes {
                if raw.len()>8192{return Err("Qwen preload plan list too large".into());}
                let shapes:Vec<(i32,i64)>=serde_json::from_str(&raw)?;
                if shapes.len()>48{return Err("too many Qwen preload plans".into());}
                if shapes.iter().any(|(kind,sequence)|!(0..=1).contains(kind)||!(1..=262144).contains(sequence)){
                    return Err("invalid Qwen preload plan shape".into());
                }
                let shapes=shapes.into_iter().collect::<std::collections::BTreeSet<_>>().into_iter()
                    .take(self.cache.capacity-3).map(|(kind,rows)|(kind,rows as usize)).collect::<Vec<_>>();
                for &(kind,sequence) in &shapes {
                    let mut bytes=0;native_error(unsafe{q3_attention_workspace(kind,sequence as i64,&mut bytes)},q3_attention_error)?;
                    prepared_attention_plans+=1;
                }
                self.cache.register_preloaded_attention(&shapes)?;
            }
        }
        self.weights_end=model.parameter_boundary()?;
        self.attach(&mut model)?;
        self.identity=server_identity(&model)?;self.plan_generation=1;
        self.fixed=Some(json!({"model_root":root,"model_manifest_sha256":sha}));self.model=Some(model);
        self.initial_load=started.elapsed().as_secs_f64();
        Ok(json!({"component":"conditioner","model_load_count":1,"requests_completed":0,
            "load_seconds":self.initial_load,"weights_identity_sha256":self.identity,
            "prepared_attention_plans":prepared_attention_plans,"forward_executed":false,
            "parallel":self.group.as_ref().map(|g|json!({"world":g.world,"rank":g.rank}))}))
    }
    fn request(&mut self,args:&[String])->Result<Value> {
        if args.first().is_some_and(|v|v=="preload") {return self.preload(args);}
        if args.first().is_some_and(|v|matches!(v.as_str(),"prepare-offload"|"offload"|"restore")) {return self.move_weights(args);}
        if self.offloaded {return Err("conditioner weights are offloaded; restore before inference".into());}
        if args.first().is_some_and(|v|v=="infer-ref2va") {return self.request_ref2va(args);}
        use crate::dynamic_request::{Request,checked_model,validate_prepared};
        crate::debug_audit::begin()?;
        let started=Instant::now();self.in_gpu=false;
        let request=Request::parse(args)?;
        let (index,manifest_sha)=checked_model(&request.model)?;
        let fixed=json!({"model_root":request.model,"model_manifest_sha256":manifest_sha});
        if self.fixed.as_ref().is_some_and(|previous|previous!=&fixed){return Err("resident conditioner model identity changed".into());}
        let output=&request.output;
        fs::create_dir_all(output)?;audit(output,"start")?;
        let admission_seconds=started.elapsed().as_secs_f64();
        let prepared=output.join("prepared");let prep_start=Instant::now();
        let (mut prep,pixels)=crate::prepare::plan_with_size(&request.model,&request.prompt,&request.canvases.iter().map(|p|p.as_path()).collect::<Vec<_>>(),&prepared,request.width,request.height)?;
        let text_rows=validate_prepared(&request,&prepared,&prep,&index,&pixels)?;
        let layout=Layout{grid_h:prep["grid_height"].as_u64().ok_or("grid height")? as usize,
            grid_w:prep["grid_width"].as_u64().ok_or("grid width")? as usize,text_rows,keyframe_count:prep["keyframe_count"].as_u64().ok_or("keyframe count")? as usize};
        // A parallel rank prepares only the pixel rows it embeds; one process, or
        // rank 0 alone below the parallel length, writes the whole tensor.
        let role=self.group.as_ref().map_or(Role::Single,|g|g.role(text_rows));
        let grids=vec![(layout.grid_h,layout.grid_w);layout.keyframe_count];
        let local_pixels=match (role,self.group.as_ref()) {
            (Role::Parallel,Some(group))=>Some(pixels.rows(vision_slice(&grids,group.world,group.rank))?),
            (Role::Single,_)=>{
                crate::prepare::write_pixels(&prepared,&mut prep,"pixel_values.raw",&pixels)?;
                if crate::debug_audit::enabled(){fs::write(prepared.join("prepared.json"),serde_json::to_vec_pretty(&prep)?)?;}
                None
            }
            _=>None,
        };
        let scratch_estimate=crate::dynamic_request::resource_limits(layout.grid_h,layout.grid_w,text_rows)?;
        fs::copy(request.model.join("model_config/config.json"),output.join("model-config.json"))?;
        let preprocess_seconds=prep_start.elapsed().as_secs_f64();
        // Token tags are final here; the pipeline may prebuild the DiT layout
        // from them while this request runs Qwen on the GPU.
        fs::write(output.join("prepared.ready"),text_rows.to_string())?;
        // All ordinary validation errors occur above, before any CUDA calls or
        // model/workspace mutation. Below, a GPU/library failure stops serving.
        // Query memory without allocating or mutating live model/plan state.
        // The explicit scratch budget rejects on CPU; device capacity adds a
        // conservative vendor-workspace reserve before beginning execution.
        let setup=Instant::now();
        let startup_preparation=self.adopt_startup_backend()?;
        let (mut free,mut total)=(0usize,0usize);cuda(unsafe{cudaMemGetInfo(&mut free,&mut total)})?;
        let weights=if self.model.is_none(){self.initial_weight_bytes(&index)?}else{0};
        let reserve=crate::dynamic_request::budget("H3_QWEN_VENDOR_RESERVE_BYTES",2usize<<30)?;
        // Sharded ranks load together even when only rank 0 computes this request;
        // an idle rank that loads needs room for its weights, not for scratch.
        let joins_load=self.text_shards.is_some() && self.model.is_none();
        let required=crate::dynamic_request::device_memory_required(role!=Role::Idle||joins_load,
            if role==Role::Idle{0}else{scratch_estimate},weights,reserve)?;
        let admitted=required<=free;
        if !admitted {eprintln!("conditioner memory admission rejected: rank={} role={role:?} free_bytes={free} required_bytes={required} scratch_bytes={scratch_estimate} weights_bytes={weights} reserve_bytes={reserve}",self.group.as_ref().map_or(0,|g|g.rank));}
        self.admit_memory(admitted,"insufficient device capacity for conditioner scratch, initial weights and vendor reserve")?;
        if role==Role::Idle && !joins_load {return self.idle_reply(started,text_rows);}
        let parallel=role==Role::Parallel;
        let world=self.group.as_ref().map_or(1,|g|g.world);
        let decision=(role!=Role::Idle).then(||self.cache.admit_images_for_world(layout.vision_sequence(),text_rows,layout.keyframe_count,if parallel{world}else{1}));
        self.in_gpu=true;
        unsafe{h3_linear_fallback_reset();}
        let mut load_seconds=0.0;
        if self.model.is_none() {
            let loading=Instant::now();
            // A rank group loads weights only; reconfigure then builds its head-group plans.
            let mut model=Model::load(&request.model,None,if role==Role::Single{layout}else{Layout::WEIGHTS_ONLY},self.text_shards)?;model.arena.sync()?;
            load_seconds=loading.elapsed().as_secs_f64();self.initial_load=load_seconds;
            self.weights_end=model.parameter_boundary()?;
            self.attach(&mut model)?;
            self.identity=server_identity(&model)?;self.plan_generation=1;
            self.model=Some(model);self.fixed=Some(fixed.clone());
        }
        if role==Role::Idle {self.in_gpu=false;return self.idle_reply(started,text_rows);}
        let decision=decision.ok_or("conditioner plan admission missing")?;
        let model=self.model.as_mut().ok_or("resident model missing")?;
        let (layout_changed,text_pending)=reconfigure(model,layout,self.weights_end,decision.clear,layout.keyframe_count>0||parallel,parallel.then_some(world))?;
        if layout_changed{self.plan_generation+=1;}
        let persistent_allocations=self.weights_end+2;
        if model.arena.allocations.len()+usize::from(text_pending)!=persistent_allocations{return Err("unreleased request allocations".into());}
        let before_forward=server_memory(model)?;
        let setup_seconds=setup.elapsed().as_secs_f64()-load_seconds;
        let forward_start=Instant::now();
        let mut phases=Phases::new(model.arena.stream);phases.mark()?;
        // The first decoder blocks copy while the vision tower runs.
        if let Some(text)=&mut model.text {text.prefetch()?;}
        let plan=match (parallel,self.group.as_ref()) {(true,Some(group))=>Some(model.plan_text_parallel(group,&prepared)?),_=>None};
        let vision=match (layout.keyframe_count,parallel,self.group.as_mut()) {
            (0,_,_)=>None,
            (_,true,Some(group))=>Some(model.vision_parallel(group,&grids,local_pixels.as_deref().ok_or("rank pixel rows missing")?)?),
            _=>Some(model.vision(&prepared)?),
        };
        // The vision tower is queued; build the text plan on the host meanwhile.
        if text_pending{
            if parallel{attach_parallel_text_workspace(model,self.weights_end,world)?;}else{attach_text_workspace(model,self.weights_end)?;}
        }
        phases.mark()?;
        let hidden=match (plan,self.group.as_mut()) {
            (Some(plan),Some(group))=>model.text_parallel(group,plan,vision,None)?,
            _=>model.text(&prepared,vision)?,
        };
        phases.mark()?;
        model.arena.sync()?;
        let forward_seconds=forward_start.elapsed().as_secs_f64();
        let gpu_phases=phases.seconds()?;drop(phases);
        let profile=match self.group.as_mut(){Some(group)=>group.take_profile()?,None=>Value::Null};
        let after_forward=server_memory(model)?;
        let handoff=Instant::now();
        // Every rank holds the gathered rows; rank 0 alone delivers them.
        let leader=self.group.as_ref().is_none_or(|g|g.rank==0);
        let conditioning=if leader{model.arena.read(hidden)?}else{vec![]};
        let download_seconds=handoff.elapsed().as_secs_f64();
        // BF16 is nonfinite exactly when its exponent bits are all ones.
        let nonfinite=conditioning.chunks_exact(2).filter(|b|u16::from_le_bytes([b[0],b[1]])&0x7f80==0x7f80).count();
        if nonfinite!=0{return Err("nonfinite resident conditioning".into());}
        let nonfinite_scan_seconds=handoff.elapsed().as_secs_f64()-download_seconds;
        if leader {
            fs::write(output.join("conditioning.raw"),&conditioning)?;
            fs::copy(prepared.join("text_token_tags.raw"),output.join("text_token_tags.raw"))?;
        }
        let write_seconds=handoff.elapsed().as_secs_f64()-download_seconds-nonfinite_scan_seconds;
        let mut plans=vec![];
        for kind in 0..2 {
            if kind==0 && layout.keyframe_count==0 {continue;}
            if parallel {plans.push(parallel_plan_name(kind,world,layout.sequence(kind as usize) as usize)?);continue;}
            let name=unsafe{q3_attention_plan_name(kind,layout.sequence(kind as usize))};
            if name.is_null(){return Err("attention plan name failed".into());}
            plans.push(unsafe{CStr::from_ptr(name)}.to_string_lossy().into_owned());
        }
        let cleanup=Instant::now();
        let (temporary_count,temporary_bytes)=release_after(model,persistent_allocations)?;
        let cleanup_seconds=cleanup.elapsed().as_secs_f64();
        let finalize=Instant::now();
        if server_identity(model)?!=self.identity{return Err("resident parameter/handle identity changed".into());}
        let memory=server_memory(model)?;audit(output,"complete")?;
        let attention_entries=unsafe{q3_attention_cache_size()}+unsafe{q3_cp_attention_cache_size()};let conv_entries=unsafe{q3_conv_cache_size(model.conv)};
        if attention_entries+conv_entries!=self.cache.entries.len(){return Err("actual plan cache differs from admitted keys".into());}
        let mut fallback_counts=[0u64;5];if unsafe{h3_linear_fallback_counts(fallback_counts.as_mut_ptr(),5)}!=0{return Err("Linear fallback counters".into());}
        let cache_report=self.cache.record(&decision);
        self.requests+=1;
        let wall=started.elapsed().as_secs_f64();let finalize_seconds=finalize.elapsed().as_secs_f64();
        let details=json!({"component":"conditioner","requests_completed":self.requests,"requests_rejected":self.rejected,
            "model_load_count":1,"request_load_seconds":load_seconds,"initial_model_load_seconds":self.initial_load,
            "request_wall_seconds":wall,"memory":memory,"preprocess_seconds":preprocess_seconds,
            "forward_seconds":forward_seconds,"scratch_cleanup_seconds":cleanup_seconds,
            "startup_backend_preparation":startup_preparation,
            "stream_phase_seconds":{"vision":gpu_phases[0],"text":gpu_phases[1]},"profile":profile,
            // Where the wall time outside preprocess/forward/cleanup goes.
            "untimed_breakdown_seconds":{"admission":admission_seconds,"setup_and_reconfigure":setup_seconds,
                "download":download_seconds,"nonfinite_scan":nonfinite_scan_seconds,"write":write_seconds,
                "finalize":finalize_seconds},
            "temporary_bytes_freed":temporary_bytes,"temporary_allocations_freed":temporary_count,
            "weights_and_handles_identity_sha256":self.identity,"debug_audit":crate::debug_audit::record(),
            "memory_before_forward":before_forward,"memory_after_forward":after_forward,
            "dynamic_request":{"enabled":true,"width":request.width,"height":request.height,"keyframe_count":layout.keyframe_count,
                "model_max_text_rows":crate::dynamic_request::MODEL_MAX_TEXT_ROWS,
                "scratch_estimated_bytes":scratch_estimate,"scratch_limit_bytes":crate::dynamic_request::budget("H3_QWEN_SCRATCH_LIMIT_BYTES",16usize<<30)?,
                "processor_width":prep["processor_width"],"processor_height":prep["processor_height"],
                "plan_cache":cache_report,"actual_attention_cache_entries":attention_entries,"actual_conv_cache_entries":conv_entries,
                "linear_fallback_counts":fallback_counts,"linear_fallback_counter_order":["lt_success","forced","dimension1","empty_heuristic","lt_failure"],
                "grid_h":layout.grid_h,"grid_w":layout.grid_w,"text_rows":text_rows,
                "layout_changed":layout_changed,"attention_plan_generation":self.plan_generation,
                "attention_workspace_bytes":model.attention_workspace.iter().map(|b|b.bytes).collect::<Vec<_>>(),
                "old_attention_plans_cleared_on_layout_change":false,"scratch_released":true,
                "parameter_count":model.params.len(),"parameter_bytes":model.params.values().map(|b|b.bytes).sum::<usize>(),
                "persistent_arena_allocations":persistent_allocations,"cuda_allocations_total":DYNAMIC_ALLOCATIONS.load(Ordering::Relaxed),
                "model_integrity_verified_at_load":true,"config_integrity_verified_before_gpu":true},
            "parallel":self.group.as_ref().map(|g|g.report(role,text_rows)),"text_shards":model.text.as_ref().map(|t|t.report())});
        let report=json!({"schema":"h3-native-conditioner-v1","passed":true,"delivered":leader,
            "native_process_clean":if crate::debug_audit::enabled(){json!(true)}else{Value::Null},"debug_audit":crate::debug_audit::record(),
            "gold_activations_read_for_computation":false,"validation_mode":false,
            "model_manifest_sha256":fixed["model_manifest_sha256"],"source_export_manifest_sha256":model.index["source_manifest_sha256"],
            "selected_hidden_state_index":50,"conditioning":{"file":"conditioning.raw","dtype":"bfloat16",
                "shape":[1,text_rows,5120],"bytes":conditioning.len(),"sha256":crate::debug_audit::sha(||hash(&conditioning)),"nonfinite":nonfinite},
            "text_token_tags":{"file":"text_token_tags.raw","dtype":"int64","shape":[text_rows],
                "sha256":if crate::debug_audit::enabled(){json!(hash(&crate::startup_trace::read(output.join("text_token_tags.raw"))?))}else{Value::Null}},
            "preparation":prep,"conv_plan":model.conv_plan,"attention_plans":plans,
            "checkpoint_count":0,"load_seconds":load_seconds,"wall_seconds":wall,"resident":details});
        fs::write(output.join("report.json"),serde_json::to_vec_pretty(&report)?)?;
        self.in_gpu=false;
        Ok(details)
    }
}

include!("resident_ref2va.rs");
include!("parallel.rs");
include!("text_shards.rs");
include!("offload.rs");

impl ConditionerServer {
    /// Short sequences run unchanged on rank 0; the other ranks only agree on admission.
    fn idle_reply(&mut self,started:Instant,text_rows:usize)->Result<Value> {
        self.requests+=1;
        Ok(json!({"component":"conditioner","requests_completed":self.requests,"requests_rejected":self.rejected,
            "model_load_count":usize::from(self.model.is_some()),"request_wall_seconds":started.elapsed().as_secs_f64(),
            "weights_and_handles_identity_sha256":self.identity,
            "parallel":self.group.as_ref().map(|g|g.report(Role::Idle,text_rows))}))
    }
}

pub fn serve(socket:&Path)->Result<()> {
    use std::{io::{BufRead,BufReader,Read,Write},os::unix::net::UnixListener};
    if socket.exists(){return Err("socket path already exists".into());}
    if let Some(parent)=socket.parent(){fs::create_dir_all(parent)?;}
    let group=Group::from_environment()?;
    // Every rank group shards the text decoder; a single process has no peers and keeps every weight.
    let text_shards=group.as_ref().map(|group|crate::text_shards::Layout::new(group.world,group.rank)).transpose()?;
    let listener=UnixListener::bind(socket)?;
    eprintln!("dynamic conditioner ready {}",socket.display());
    let mut server=ConditionerServer{group,model:None,fixed:None,identity:String::new(),requests:0,rejected:0,
        initial_load:0.0,weights_end:0,plan_generation:0,in_gpu:false,cache:crate::plan_cache::Cache::from_env()?,text_shards,
        host_weights:None,offloaded:false,gpu_restore_count:0};
    for connection in listener.incoming() {
        let mut stream=connection?;server.in_gpu=false;
        let result=(||->Result<Value>{
            let mut line=String::new();
            BufReader::new((&stream).take((1<<20)+1)).read_line(&mut line)?;
            if line.len()>1<<20 || !line.ends_with('\n'){return Err("invalid newline-delimited request".into());}
            let value:Value=serde_json::from_str(&line)?;
            let args:Vec<String>=serde_json::from_value(value["args"].clone())?;
            server.request(&args)
        })();
        let fatal=server.in_gpu && result.is_err();
        let reply=match result {
            Ok(details)=>json!({"ok":true,"error":Value::Null,"details":details}),
            Err(error)=>{
                server.rejected+=1;
                json!({"ok":false,"error":error.to_string(),"details":{"component":"conditioner",
                    "requests_completed":server.requests,"requests_rejected":server.rejected,
                    "model_load_count":usize::from(server.model.is_some()),"recoverable":!fatal,
                    "gpu_execution_started":server.in_gpu,"cuda_allocations_total":DYNAMIC_ALLOCATIONS.load(Ordering::Relaxed),
                    "weights_and_handles_identity_sha256":server.identity}})
            },
        };
        writeln!(stream,"{}",serde_json::to_string(&reply)?)?;stream.flush()?;
        if fatal{return Err("dynamic conditioner stopped after GPU/execution error".into());}
    }
    Ok(())
}
