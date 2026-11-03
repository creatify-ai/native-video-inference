//! Case-01 video VAE orchestration. Runtime inputs are live RGB/noise/latents;
//! reference tensors are read only by the explicit postquant fixture gate.
use super::*;
#[path="geometry.rs"]
mod geometry;

unsafe extern "C" {
    fn h3v_encode_prepare(rgb:*const u8,pixels:*mut f32,channels_last:i32,stream:Stream)->i32;
    fn h3v_extract_f32(input:*const f32,tile:*mut f32,c:i32,t:i32,h:i32,w:i32,t0:i32,y0:i32,x0:i32,tt:i32,th:i32,tw:i32,channels_last:i32,stream:Stream)->i32;
    fn h3v_stitch(tiles:*const c_void,output:*mut c_void,c:i32,t:i32,k:i32,dtype:i32,stream:Stream)->i32;
    fn h3v_decode_denorm(input:*const f32,mean:*const f32,std:*const f32,out:*mut f32,stream:Stream)->i32;
    fn h3v_posterior(moments:*const f32,noise:*const f32,mean:*const f32,std:*const f32,out:*mut f32,stream:Stream)->i32;
    fn h3v_temporal_merge(clip:*const c_void,overlap:*mut c_void,out:*mut c_void,chunk:i32,stream:Stream)->i32;
    fn h3v_pt8(input:*const c_void,output:*mut u8,stream:Stream)->i32;
    fn h3v_half_nonfinite(input:*const c_void,counter:*mut u32,count:i64,stream:Stream)->i32;
    fn h3v_cast_half(input:*const c_void,output:*mut c_void,n:i64,stream:Stream)->i32;
    fn h3v_bias_half(input:*mut c_void,bias:*const c_void,n:i64,spatial:i64,c:i64,stream:Stream)->i32;
    fn h3vg_encode_prepare(input:*const u8,out:*mut f32,h:i32,w:i32,stream:Stream)->i32;
    fn h3vg_extract(input:*const f32,out:*mut f32,c:i32,t:i32,h:i32,w:i32,t0:i32,y0:i32,x0:i32,tt:i32,th:i32,tw:i32,stream:Stream)->i32;
    fn h3vg_stitch(input:*const c_void,out:*mut c_void,c:i32,t:i32,k:i32,h:i32,w:i32,ny:i32,nx:i32,ys:*const i32,xs:*const i32,yo:*const i32,xo:*const i32,dtype:i32,stream:Stream)->i32;
    fn h3vgs_stitch(input:*const c_void,out:*mut c_void,c:i32,t:i32,kh:i32,kw:i32,h:i32,w:i32,ny:i32,nx:i32,ys:*const i32,xs:*const i32,yo:*const i32,xo:*const i32,dtype:i32,stream:Stream)->i32;
    fn h3vg_denorm(input:*const f32,mean:*const f32,std:*const f32,out:*mut f32,spatial_temporal:i64,stream:Stream)->i32;
    fn h3vg_posterior(input:*const f32,noise:*const f32,mean:*const f32,std:*const f32,out:*mut f32,spatial:i64,stream:Stream)->i32;
    fn h3vg_temporal_merge(input:*const c_void,overlap:*mut c_void,out:*mut c_void,chunk:i32,chunks:i32,h:i32,w:i32,frames:i32,stream:Stream)->i32;
    fn h3vg_pt8(input:*const c_void,out:*mut u8,h:i32,w:i32,frames:i32,stream:Stream)->i32;
}

fn read_input(path:&Path,bytes:usize)->Result<Vec<u8>>{
    let value=crate::startup_trace::read(path)?;if value.len()!=bytes{return Err(format!("runtime input size mismatch: {}",path.display()).into());}Ok(value)
}
fn allocate_upload(arena:&mut Arena,bytes:&[u8])->Result<DevicePtr>{let ptr=arena.alloc(bytes.len())?;arena.upload(ptr,bytes)?;Ok(ptr)}
fn axis_upload(arena:&mut Arena,values:&[i32],scale:i32)->Result<DevicePtr>{
    let mut bytes=Vec::new();for x in values{bytes.extend_from_slice(&(x/scale).to_le_bytes());}
    if bytes.is_empty(){bytes.extend_from_slice(&0i32.to_le_bytes());}allocate_upload(arena,&bytes)
}
struct Spatial {width:usize,height:usize,x:geometry::Axis,y:geometry::Axis}
impl Spatial{
    fn new(width:usize,height:usize)->Result<Self>{Ok(Self{width,height,x:geometry::axis(width)?,y:geometry::axis(height)?})}
    fn area(&self)->usize{self.width*self.height}
    fn latent_area(&self)->usize{self.area()/256}
    fn tiles(&self)->usize{self.x.starts.len()*self.y.starts.len()}
    fn arrays(&self,arena:&mut Arena,scale:i32)->Result<[DevicePtr;4]>{Ok([axis_upload(arena,&self.y.starts,scale)?,axis_upload(arena,&self.x.starts,scale)?,axis_upload(arena,&self.y.overlaps,scale)?,axis_upload(arena,&self.x.overlaps,scale)?])}
}
fn floats(values:&Value)->Result<Vec<u8>>{let mut result=Vec::new();for v in arr(values)?{result.extend_from_slice(&(v.as_f64().ok_or("invalid channel constant")? as f32).to_le_bytes());}if result.len()!=24*4{return Err("video VAE expects 24 channel constants".into());}Ok(result)}
fn finite_f32(bytes:&[u8])->bool{bytes.chunks_exact(4).all(|v|f32::from_le_bytes(v.try_into().unwrap()).is_finite())}

/// Ref2VA temporal reference encoding. This stage executes on rank zero;
/// the same conditioning is subsequently consumed by single-GPU or CP8 DiT.
struct TemporalEncoder { root:PathBuf, program:Program, mean:DevicePtr, std:DevicePtr, persistent_mark:usize, persistent_bytes:usize, requests:usize, load_seconds:f64 }
impl TemporalEncoder {
    fn load(root:&Path)->Result<Self> {
        let started=Instant::now();let root=fs::canonicalize(root)?;
        let _trace=crate::startup_trace::Span::new("submodel","reference_encoder",0);
        let mut program=Program::load_execution(&root,false)?;
    if program.manifest["stage"]!="video_vae_encoder_clip17_tile" || program.manifest["outputs"][0]["shape"]!=json!([1,48,5,16,16]) {
        return Err("requires original 17-frame encoder execution plan".into());
    }
    let inputs:Vec<_>=arr(&program.manifest["buffers"])?.iter().filter(|b|b["role"]=="input").collect();
    if inputs.len()!=1 || inputs[0]["shape"]!=json!([1,3,17,256,256]){return Err("temporal tile input contract".into());}
    let audit_root=Path::new(text(&program.manifest["audit_root"])?);
    let audit:Value=serde_json::from_slice(&crate::startup_trace::read(audit_root.join("manifest.json"))?)?;
        let mean=allocate_upload(&mut program.arena,&floats(&audit["config"]["latents_mean"])?)?;
        let std=allocate_upload(&mut program.arena,&floats(&audit["config"]["latents_std"])?)?;

        program.arena.sync()?;
        let persistent_mark=program.arena.allocation_mark();let persistent_bytes=program.arena.allocation_bytes();
        Ok(Self{root,program,mean,std,persistent_mark,persistent_bytes,requests:0,load_seconds:started.elapsed().as_secs_f64()})
    }
    fn execute(&mut self,args:&[String])->Result<Value> {
        crate::debug_audit::begin()?;
        if args.len()!=9 || fs::canonicalize(&args[2])?!=self.root{return Err("temporal encoder arguments/model identity".into());}
        if self.program.arena.allocation_mark()!=self.persistent_mark{return Err("unreleased temporal encoder allocations".into());}
        self.program.ensure_workspace()?;
        let outcome=encode_video_body(args,self);
        let peak=self.program.arena.allocation_bytes();
        self.program.arena.release_after(self.persistent_mark)?;
        self.program.workspace_released();
        if self.program.arena.allocation_bytes()!=self.persistent_bytes{return Err("temporal encoder scratch release mismatch".into());}
        let mut report=outcome?;self.requests+=1;
        report["resident"]=json!({"component":"video_reference_encoder","model_load_count":1,
            "requests_completed":self.requests,"request_load_seconds":if self.requests==1{self.load_seconds}else{0.},
            "persistent_bytes":self.persistent_bytes,"request_peak_bytes":peak,"temporary_allocations_released":true});
        fs::write(Path::new(&args[5]).join("report.json"),serde_json::to_vec_pretty(&report)?)?;
        Ok(report)
    }
}

pub(super) fn encode_video(args:&[String])->Result<()> {
    if args.len()!=9{return Err("encode-video PROGRAM_ROOT RGB_THWC NOISE42 OUTPUT WIDTH HEIGHT FRAMES".into());}
    let mut encoder=TemporalEncoder::load(Path::new(&args[2]))?;
    println!("{}",encoder.execute(args)?);Ok(())
}

fn encode_video_body(args:&[String],ctx:&mut TemporalEncoder)->Result<Value> {
    if args.len()!=9{return Err("encode-video PROGRAM_ROOT RGB_THWC NOISE42 OUTPUT WIDTH HEIGHT FRAMES".into());}
    crate::debug_audit::begin()?;
    let started=Instant::now();let root=Path::new(&args[2]);let out=Path::new(&args[5]);
    if !out.is_absolute() || out.exists(){return Err("fresh absolute video-reference output required".into());}
    let (w,h,f):(usize,usize,usize)=(args[6].parse()?,args[7].parse()?,args[8].parse()?);
    if w<256 || h<256 || !(13..=345).contains(&f){return Err("temporal encoder requires axes >=256 and 13..345 normalized frames".into());}
    let shape=Spatial::new(w,h)?;let used=f.min(((f-5)/17).max(1)*17+5);
    let chunks=used.div_ceil(17);let latent_frames=chunks*5-3;let spatial=shape.latent_area();
    let rgb=read_input(Path::new(&args[3]),f.checked_mul(shape.area()).and_then(|v|v.checked_mul(3)).ok_or("video bytes overflow")?)?;
    let noise=read_input(Path::new(&args[4]),24*latent_frames*spatial*4)?;
    if !finite_f32(&noise){return Err("nonfinite posterior noise".into());}
    let program=&mut ctx.program;let mean=ctx.mean;let std=ctx.std;
    let input=allocate_upload(&mut program.arena,&rgb[..used*shape.area()*3])?;
    let noise_gpu=allocate_upload(&mut program.arena,&noise)?;
    let pixels=program.arena.alloc(3*used*shape.area()*4)?;
    // This per-pixel kernel flattens H*W, so [T*H,W] maps THWC -> CTHW
    // with exactly the already verified ImageNet normalization arithmetic.
    check(unsafe{h3vg_encode_prepare(input as _,pixels as _,(used*h) as _,w as _,program.arena.stream)})?;
    let tile_in=input_pointer(&program)?;let tile_out=output_pointer(&program)?;
    let tiles=program.arena.alloc(shape.tiles()*48*5*16*16*4)?;
    let stitched=program.arena.alloc(48*5*spatial*4)?;
    let moments=program.arena.alloc(48*latent_frames*spatial*4)?;
    let condition=program.arena.alloc(24*latent_frames*spatial*4)?;
    let [ys,xs,yo,xo]=shape.arrays(&mut program.arena,16)?;
    for chunk in 0..chunks {
        for (row,y) in shape.y.starts.iter().enumerate(){for (col,x) in shape.x.starts.iter().enumerate(){
            check(unsafe{h3vg_extract(pixels as _,tile_in as _,3,used as _,h as _,w as _,(chunk*17) as _,*y,*x,17,256,256,program.arena.stream)})?;
            program.run(false)?;
            check(unsafe{cuMemcpyDtoDAsync_v2(tiles+((row*shape.x.starts.len()+col)*48*5*16*16*4) as u64,tile_out,48*5*16*16*4,program.arena.stream)})?;
        }}
        check(unsafe{h3vgs_stitch(tiles as _,stitched as _,48,5,16,16,(h/16) as _,(w/16) as _,shape.y.starts.len() as _,shape.x.starts.len() as _,ys as _,xs as _,yo as _,xo as _,0,program.arena.stream)})?;
        let keep=(latent_frames-chunk*5).min(5);
        for channel in 0..48 {
            check(unsafe{cuMemcpyDtoDAsync_v2(moments+((channel*latent_frames+chunk*5)*spatial*4) as u64,
                stitched+(channel*5*spatial*4) as u64,keep*spatial*4,program.arena.stream)})?;
        }
    }
    check(unsafe{h3vg_posterior(moments as _,noise_gpu as _,mean as _,std as _,condition as _,(latent_frames*spatial) as _,program.arena.stream)})?;
    let bytes=program.arena.download(condition,24*latent_frames*spatial*4)?;
    if !finite_f32(&bytes){return Err("nonfinite video-reference conditioning".into());}
    fs::create_dir(out)?;fs::write(out.join("condition.raw"),&bytes)?;
    let clean=maps(out)?;
    let report=json!({"passed":clean,"mode":"encode-video","input_shape":[f,h,w,3],"used_frames":used,
        "shape":[1,24,latent_frames,h/16,w/16],"sha256":hash(&bytes),"bytes":bytes.len(),"nonfinite":0,
        "posterior_seed_input_sha256":hash(&noise),"gold_activations_read":false,"execution_world":1,
        "program_sha256":hash(&crate::startup_trace::read(root.join("program.json"))?),"wall_seconds":started.elapsed().as_secs_f64()});
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&report)?)?;
    if !clean{return Err("forbidden native dependency".into());}Ok(report)
}
fn input_pointer(program:&Program)->Result<DevicePtr>{
    let specs=arr(&program.manifest["buffers"])?;let ids:Vec<_>=specs.iter().enumerate().filter(|(_,v)|v["role"]=="input").map(|(i,_)|i).collect();
    if ids.len()!=1{return Err("component requires one declared runtime input".into());}Ok(program.buffers[ids[0]])
}
fn output_pointer(program:&Program)->Result<DevicePtr>{program.pointer(&program.manifest["outputs"][0])}
fn maps(out:&Path)->Result<bool>{if !crate::debug_audit::enabled(){return Ok(true);}let value=fs::read_to_string("/proc/self/maps")?;fs::write(out.join("native.maps"),&value)?;Ok(!["libtorch","libpython","libaten","libc10"].iter().any(|x|value.to_lowercase().contains(x)))}

struct Model { manifest:Value,audit:Value,audit_root:PathBuf,program_root:PathBuf }
impl Model{
    fn load(root:&Path,stage:&str)->Result<Self>{
        let manifest:Value=serde_json::from_slice(&crate::startup_trace::read(root.join("model.json"))?)?;
        if manifest["schema"]!="h3_native_video_vae_v1"{return Err("unsupported video model package".into());}
        let audit_root=PathBuf::from(text(&manifest["audit_root"])?);let raw=crate::startup_trace::read(audit_root.join("manifest.json"))?;
        if hash(&raw)!=text(&manifest["audit_manifest_sha256"])?{return Err("video audit changed".into());}
        let audit=serde_json::from_slice(&raw)?;let entry=&manifest[stage];let program_root=PathBuf::from(text(&entry["root"])?);
        if hash(&crate::startup_trace::read(program_root.join("program.json"))?)!=text(&entry["program_sha256"])?{return Err("video execution plan changed".into());}
        Ok(Self{manifest,audit,audit_root,program_root})
    }
}

struct PostQuant { ctx:*mut c_void,w:DevicePtr,b:DevicePtr,input:DevicePtr }
impl PostQuant {
    fn new(arena:&mut Arena,audit_root:&Path,audit:&Value)->Result<Self>{
        let mut ctx=ptr::null_mut();if unsafe{h3_conv_create(&mut ctx)}!=0{return Err(unsafe{CStr::from_ptr(h3_conv_error())}.to_string_lossy().into_owned().into());}
        let mut half=Vec::new();
        for name in ["post_quant_conv.weight","post_quant_conv.bias"]{
            let spec=&audit["parameters"][name];if spec["dtype"]!="torch.float32"{return Err("postquant must retain FP32 weights with autocast".into());}
            let bytes=read_model_checked(audit_root,spec)?;let fp32=allocate_upload(arena,&bytes)?;let fp16=arena.alloc(bytes.len()/2)?;
            check(unsafe{h3v_cast_half(fp32 as _,fp16 as _,(bytes.len()/4) as _,arena.stream)})?;half.push(fp16);
        }
        let input=arena.alloc(24*7*16*16*2)?;Ok(Self{ctx,w:half[0],b:half[1],input})
    }
    fn forward(&self,arena:&Arena,input:DevicePtr,output:DevicePtr)->Result<()> {
        self.forward_spatial(arena,input,output,16,16)
    }
    fn forward_spatial(&self,arena:&Arena,input:DevicePtr,output:DevicePtr,h:usize,w:usize)->Result<()> {
        let (h,w)=(h as i64,w as i64);
        check(unsafe{h3v_cast_half(input as _,self.input as _,24*7*h*w,arena.stream)})?;
        let spec=ConvSpec{ndim:5,dtype:1,allow_tf32:1,input_dims:[1,24,7,h,w],input_strides:[24*7*h*w,7*h*w,h*w,w,1],
            weight_dims:[24,24,1,1,1],weight_strides:[24,1,1,1,1],output_dims:[1,24,7,h,w],output_strides:[24*7*h*w,7*h*w,h*w,w,1],
            stride:[1,1,1],dilation:[1,1,1],..Default::default()};
        if unsafe{h3_conv_forward(self.ctx,&spec,self.input as _,self.w as _,output as _,arena.stream)}!=0{return Err(unsafe{CStr::from_ptr(h3_conv_error())}.to_string_lossy().into_owned().into());}
        check(unsafe{h3v_bias_half(output as _,self.b as _,24*7*h*w,7*h*w,24,arena.stream)})
    }
}
impl Drop for PostQuant{fn drop(&mut self){unsafe{h3_conv_destroy(self.ctx);}}}

fn encode_body(args:&[String],ctx:&mut Context,started:Instant,model_load_seconds:f64)->Result<Value> {
    let world=TileTeam::configured_world()?;
    if ctx.tile_team.is_none(){ctx.tile_team=Some(TileTeam::new(world)?);}
    if args.len()!=7&&args.len()!=9{return Err("usage: h3-video-vae encode MODEL_ROOT FIRST_RGB LAST_RGB NOISE42_RAW OUTPUT_DIR [WIDTH HEIGHT]".into());}
    let model=&ctx.model;let program=&mut ctx.program;let (mean,std)=(ctx.mean,ctx.std);let out=Path::new(&args[6]);fs::create_dir_all(out)?;
    let (width,height)=if args.len()==9{(args[7].parse()?,args[8].parse()?)}else{(768,768)};
    let shape=Spatial::new(width,height)?;let lh=height/16;let lw=width/16;let spatial=shape.latent_area();
    let (th,tw)=(height.min(256),width.min(256));let (kh,kw)=(th/16,tw/16);
    program.select_spatial("encoder",th,tw)?;
    let tile_in=input_pointer(program)?;let tile_out=output_pointer(program)?;
    let noise_bytes=read_input(Path::new(&args[5]),24*spatial*4)?;if !finite_f32(&noise_bytes){return Err("nonfinite encoder noise".into());}
    let noise=allocate_upload(&mut program.arena,&noise_bytes)?;
    let rgb=program.arena.alloc(shape.area()*3)?;let pixels=program.arena.alloc(3*shape.area()*4)?;
    let tiles=program.arena.alloc(shape.tiles()*48*kh*kw*4)?;let moments=program.arena.alloc(48*spatial*4)?;let condition=program.arena.alloc(24*spatial*4)?;
    let [ys,xs,yo,xo]=shape.arrays(&mut program.arena,16)?;
    let preparation_seconds=started.elapsed().as_secs_f64();let mut results=Vec::new();
    let keyframes=if args[4]=="-" {1}else{2};
    for image in 0..keyframes{
        let bytes=read_input(Path::new(&args[3+image]),shape.area()*3)?;program.arena.upload(rgb,&bytes)?;
        check(unsafe{h3vg_encode_prepare(rgb as _,pixels as _,height as _,width as _,program.arena.stream)})?;
        program.arena.sync()?;let gpu_context=current_context()?;
        let team=ctx.tile_team.as_mut().ok_or("missing VAE encoder tile team")?;
        let prepared=team.prepare(TileJob{model_root:ctx.model_root.clone(),stage:"encoder",source:pixels,
            source_context:gpu_context,source_bytes:3*shape.area()*4,target:tiles,target_context:gpu_context,
            tile_bytes:48*kh*kw*4,width,height,latent_frames:1,th,tw,ys:shape.y.starts.clone(),xs:shape.x.starts.clone(),world,coordinator:true,slot_bytes:0});
        if let Err(error)=prepared{let _=team.finish();return Err(error);}
        let mut session=TileSession{team,finished:false};let pending=session.team.dispatch(0)?;
        let local=(||->Result<()>{
        for (row,y) in shape.y.starts.iter().enumerate(){for (col,x) in shape.x.starts.iter().enumerate(){
            if (row*shape.x.starts.len()+col)%session.team.active!=0{continue;}
            check(unsafe{h3vg_extract(pixels as _,tile_in as _,3,1,height as _,width as _,0,*y,*x,1,th as _,tw as _,program.arena.stream)})?;
            program.run(false)?;
            check(unsafe{cuMemcpyDtoDAsync_v2(tiles+((row*shape.x.starts.len()+col)*48*kh*kw*4) as u64,tile_out,48*kh*kw*4,program.arena.stream)})?;
        }}
        Ok(())})();let remote=session.team.drain(pending);local?;remote?;
        check(unsafe{h3vgs_stitch(tiles as _,moments as _,48,1,kh as _,kw as _,lh as _,lw as _,shape.y.starts.len() as _,shape.x.starts.len() as _,ys as _,xs as _,yo as _,xo as _,0,program.arena.stream)})?;
        check(unsafe{h3vg_posterior(moments as _,noise as _,mean as _,std as _,condition as _,spatial as _,program.arena.stream)})?;
        let output=program.arena.download(condition,24*spatial*4)?;if !finite_f32(&output){return Err("nonfinite encoder condition".into());}
        fs::write(out.join(format!("condition{image}.raw")),&output)?;
        session.finish()?;
        results.push(json!({"image":image,"input_sha256":crate::debug_audit::sha(||hash(&bytes)),"output_sha256":crate::debug_audit::sha(||hash(&output)),"bytes":output.len(),"finite":true}));
        if crate::debug_audit::enabled(){eprintln!("encoded keyframe {image}: {}",hash(&output));}
    }
    let clean=maps(out)?;let report=json!({"passed":clean,"mode":"encode","validation":"execution_only","no_reference_activations_read":true,
        "native_dependencies_clean":if crate::debug_audit::enabled(){json!(clean)}else{Value::Null},"debug_audit":crate::debug_audit::record(),"model":model.manifest,"noise_sha256":crate::debug_audit::sha(||hash(&noise_bytes)),"input_shape":[height,width,3],"output_shape":[1,24,1,lh,lw],"spatial_tiles":[shape.y.starts.len(),shape.x.starts.len()],"load_seconds":model_load_seconds,"preparation_seconds":preparation_seconds,"wall_seconds":started.elapsed().as_secs_f64(),"results":results,"parallel_vae":ctx.tile_team.as_ref().map(|t|t.report())});
    if crate::debug_audit::enabled(){fs::write(out.join("report.json"),serde_json::to_string_pretty(&report)?+"\n")?;}if !clean{return Err("forbidden dependency".into());}Ok(report)
}

/// Complete relay output: bounded raw-RGB batches reuse the ordinary native
/// resize and streaming-media contract.
fn deliver_rgb(args:&[String],ctx:&mut Context)->Result<Value>{
    let width:usize=args[5].parse()?;let height:usize=args[6].parse()?;let frames:usize=args[7].parse()?;
    if width==0||height==0||width>8192||height>8192||frames==0||frames>1500{return Err("invalid relay delivery geometry".into());}
    let bytes=width.checked_mul(height).and_then(|n|n.checked_mul(3)).ok_or("relay frame size overflow")?;
    let mut source=fs::File::open(&args[3])?;
    if source.metadata()?.len()!=bytes.checked_mul(frames).ok_or("relay video size overflow")? as u64{return Err("relay RGB size mismatch".into());}
    let out=Path::new(&args[4]);fs::create_dir_all(out)?;
    let request=ctx.request.as_ref().ok_or("relay delivery requires a request context")?;
    let mut stream=FrameStream::new(request,width,height,frames,out)?;
    let count=request.chunk_frames();let delivered=request.delivery.map_or(frames,|d|d[2]);
    let device=ctx.program.arena.alloc(count*bytes)?;let mut chunk=vec![0u8;count*bytes];
    for first in (0..delivered).step_by(count){
        let n=count.min(delivered-first);source.read_exact(&mut chunk[..n*bytes])?;
        ctx.program.arena.upload(device,&chunk[..n*bytes])?;
        stream.emit(device,first,n,&ctx.program.arena,height,width)?;
    }
    let report=stream.finish()?;
    Ok(json!({"passed":true,"mode":"deliver-rgb","streaming":report,"no_reference_activations_read":true}))
}

fn decode_body(args:&[String],ctx:&mut Context,started:Instant,model_load_seconds:f64)->Result<Value> {
    if args.get(1).is_some_and(|v|v=="decode-long"){return decode_long_body(args,ctx,started,model_load_seconds);}
    decode_slice_body(args,ctx,started,model_load_seconds,None)
}
fn decode_slice_body(args:&[String],ctx:&mut Context,started:Instant,model_load_seconds:f64,
                     mut long_output:Option<(&mut FrameStream,usize,usize,usize)>)->Result<Value> {
    // Opt-in (H3_VAE_DECODE_PROFILE_DIR) split of the per-request preparation before the first tile.
    let prep_profiled=std::env::var_os("H3_VAE_DECODE_PROFILE_DIR").is_some();
    let mut prep_mark=Instant::now();let mut prep_split=serde_json::Map::new();
    let mut prep_lap=|name:&str|{if prep_profiled{prep_split.insert(name.into(),json!(prep_mark.elapsed().as_secs_f64()));prep_mark=Instant::now();}};

    let world=TileTeam::configured_world()?;
    if ctx.tile_team.is_none(){ctx.tile_team=Some(TileTeam::new(world)?);}
    if args.len()!=5&&args.len()!=8{return Err("usage: h3-video-vae decode MODEL_ROOT LATENTS_RAW OUTPUT_DIR [WIDTH HEIGHT NUM_FRAMES]".into());}
    let model=&ctx.model;let program=&mut ctx.program;let (mean,std)=(ctx.mean,ctx.std);let out=Path::new(&args[4]);fs::create_dir_all(out)?;let post=ctx.post.as_ref().ok_or("decoder postquant context missing")?;
    let (width,height,frames): (usize,usize,usize)=if args.len()==8{(args[5].parse()?,args[6].parse()?,args[7].parse()?)}else{(768,768,345)};
    let shape=Spatial::new(width,height)?;let lh=height/16;let lw=width/16;let pixels=shape.area();let spatial=shape.latent_area();
    let (th,tw)=(height.min(256),width.min(256));let (kh,kw)=(th/16,tw/16);
    program.select_spatial("decoder",th,tw)?;
    crate::limits::video_elements(width,height,frames)?;
    let input_bytes=crate::startup_trace::read(Path::new(&args[3]))?;
    if input_bytes.len()%(24*spatial*4)!=0{return Err("video input is not contiguous FP32[1,24,T,H/16,W/16]".into());}
    let temporal=geometry::temporal(input_bytes.len()/(24*spatial*4))?;
    if temporal.frames!=frames{return Err(format!("original VAE temporal output is {}, requested {frames}",temporal.frames).into());}
    if !finite_f32(&input_bytes){return Err("nonfinite decoder input".into());}
    prep_lap("read_and_check_latents");
    let tile_in=input_pointer(program)?;let tile_out=output_pointer(program)?;
    let input=allocate_upload(&mut program.arena,&input_bytes)?;
    let denormalized=program.arena.alloc(input_bytes.len())?;let prequant=program.arena.alloc(24*7*kh*kw*4)?;
    // Local diagnostic capture needs the first tile to execute on the owning GPU.
    let queued=decode_queue_enabled()&&world>1&&std::env::var("H3_VAE_CAPTURE_INPUT").as_deref()!=Ok("1");
    let tile_bytes=3*28*th*tw*2;let slot_bytes=shape.tiles()*tile_bytes;
    let tiles=program.arena.alloc(if queued{QUEUE_SLOTS*slot_bytes}else{slot_bytes})?;
    let clip=program.arena.alloc(3*28*pixels*2)?;let overlap=program.arena.alloc(3*5*pixels*2)?;
    let video=program.arena.alloc(3*frames*pixels*2)?;
    let streaming=ctx.request.is_some();
    let delivery=ctx.request.as_ref().and_then(|r|r.delivery);
    let chunk_frames=ctx.request.as_ref().map_or(22,VideoRequest::chunk_frames);
    let pt8=program.arena.alloc(if streaming{chunk_frames*pixels*3}else{frames*pixels*3})?;
    prep_lap("device_allocations");
    let mut frame_stream=if long_output.is_none(){if let Some(request)=ctx.request.as_ref(){Some(FrameStream::new(request,width,height,frames,out)?)}else{None}}else{None};
    prep_lap("frame_stream");
    let [ys,xs,yo,xo]=shape.arrays(&mut program.arena,1)?;
    let nonfinite=allocate_upload(&mut program.arena,&[0,0,0,0])?;
    check(unsafe{h3vg_denorm(input as _,mean as _,std as _,denormalized as _,(spatial*temporal.latent_frames) as _,program.arena.stream)})?;
    program.arena.sync()?;
    prep_lap("denorm");
    if std::env::var("H3_VAE_CAPTURE_INPUT").as_deref()==Ok("1"){
        fs::write(out.join("vae-input.raw"),program.arena.download(denormalized,input_bytes.len())?)?;
    }
    let gpu_context=current_context()?;
    let team=ctx.tile_team.as_mut().ok_or("missing VAE tile team")?;
    let prepared=team.prepare(TileJob{model_root:ctx.model_root.clone(),stage:"decoder",source:denormalized,
        source_context:gpu_context,source_bytes:input_bytes.len(),target:tiles,target_context:gpu_context,tile_bytes,
        width,height,latent_frames:temporal.latent_frames,th,tw,ys:shape.y.starts.clone(),xs:shape.x.starts.clone(),world,coordinator:true,slot_bytes});
    if let Err(error)=prepared{let _=team.finish();return Err(error);}
    prep_lap("tile_team_prepare");
    let mut session=TileSession{team,finished:false};
    let preparation_seconds=started.elapsed().as_secs_f64();
    // Opt-in phase timing: synchronizes the stream between phases (slows the run
    // slightly); off by default. Written to H3_VAE_DECODE_PROFILE_DIR.
    let profile_dir=std::env::var_os("H3_VAE_DECODE_PROFILE_DIR").map(PathBuf::from);
    let mut phase=[0f64;5];let mut phase_mark=Instant::now();
    macro_rules! mark {($i:expr) => { if profile_dir.is_some(){program.arena.sync()?;phase[$i]+=phase_mark.elapsed().as_secs_f64();phase_mark=Instant::now();} } }
    // GPU0 stitches, merges and streams each chunk serially, in order.
    // Queued decode (default): GPU0 and the workers take tiles from one queue across chunk
    // boundaries, and GPU0 decodes tiles whenever it would otherwise wait for this chunk.
    // Fixed split (H3_VAE_DECODE_QUEUE=0): the next chunk's remote tiles are dispatched before
    // GPU0's serial work, and GPU0 computes no tiles unless that lowers the per-GPU maximum.
    // Tiles are independent, so who decodes a tile never changes a decoded pixel.
    let coordinator_tiles=session.team.coordinator_tiles()&&!queued;
    let owns_local=|index:usize,active:usize|coordinator_tiles&&index%active==0;
    let queue=queued.then(||Arc::new(DecodeQueue::new(shape.tiles(),temporal.chunks)));
    let mut pending=match &queue{Some(queue)=>session.team.start_queue(queue)?,None=>session.team.dispatch(0)?};
    // GPU0's own queued tiles, the longest single-chunk emit and when the first frames left GPU0.
    let (mut own_tiles,mut emit_max,mut first_emit)=(0usize,0f64,-1f64);
    let decoded=(||->Result<()>{
    if queue.is_some(){program.arena.upload(nonfinite,&[0,0,0,0])?;}
    for chunk in 0..temporal.chunks{
        let t=Instant::now();
        if profile_dir.is_some(){phase_mark=Instant::now();}
        let slot=tiles+(if queue.is_some(){(chunk%QUEUE_SLOTS)*slot_bytes}else{0}) as u64;
        if let Some(queue)=&queue{
            while !queue.chunk_complete(chunk)?{match queue.take()?{
                Take::Tile(c,index)=>{
                    let tiled=Instant::now();let (row,col)=(index/shape.x.starts.len(),index%shape.x.starts.len());
                    check(unsafe{h3vg_extract(denormalized as _,prequant as _,24,temporal.latent_frames as _,lh as _,lw as _,(c*5) as _,shape.y.starts[row]/16,shape.x.starts[col]/16,7,kh as _,kw as _,program.arena.stream)})?;
                    post.forward_spatial(&program.arena,prequant,tile_in,kh,kw)?;program.run(false)?;
                    check(unsafe{h3v_half_nonfinite(tile_out as _,nonfinite as _,(tile_bytes/2) as i64,program.arena.stream)})?;
                    check(unsafe{cuMemcpyDtoDAsync_v2(tiles+((c%QUEUE_SLOTS)*slot_bytes+index*tile_bytes) as u64,tile_out,tile_bytes,program.arena.stream)})?;
                    program.arena.sync()?;queue.complete(c)?;own_tiles+=1;phase[0]+=tiled.elapsed().as_secs_f64();
                },
                Take::Wait|Take::Exhausted=>{
                    let waiting=Instant::now();queue.wait_until(|q,s|s.done[chunk]==q.tiles||q.available(s))?;phase[1]+=waiting.elapsed().as_secs_f64();
                },
            }}
            if profile_dir.is_some(){phase_mark=Instant::now();}
        }else{
            program.arena.upload(nonfinite,&[0,0,0,0])?;
            let active=session.team.active;
            let local=(||->Result<()>{
            for (row,y) in shape.y.starts.iter().enumerate(){for (col,x) in shape.x.starts.iter().enumerate(){
                if !owns_local(row*shape.x.starts.len()+col,active){continue;}
                check(unsafe{h3vg_extract(denormalized as _,prequant as _,24,temporal.latent_frames as _,lh as _,lw as _,(chunk*5) as _,*y/16,*x/16,7,kh as _,kw as _,program.arena.stream)})?;
                post.forward_spatial(&program.arena,prequant,tile_in,kh,kw)?;program.run(false)?;
                if chunk==0&&row==0&&col==0&&std::env::var("H3_VAE_CAPTURE_INPUT").as_deref()==Ok("1"){
                    fs::write(out.join("decoder-tile-input.raw"),program.arena.download(tile_in,24*7*kh*kw*2)?)?;
                    fs::write(out.join("decoder-tile-output.raw"),program.arena.download(tile_out,tile_bytes)?)?;
                }
                check(unsafe{h3v_half_nonfinite(tile_out as _,nonfinite as _,(tile_bytes/2) as i64,program.arena.stream)})?;
                check(unsafe{cuMemcpyDtoDAsync_v2(tiles+((row*shape.x.starts.len()+col)*tile_bytes) as u64,tile_out,tile_bytes,program.arena.stream)})?;
            }}
            Ok(())})();
            if profile_dir.is_some()&&local.is_ok(){program.arena.sync()?;phase[0]+=phase_mark.elapsed().as_secs_f64();phase_mark=Instant::now();}
            let remote=session.team.drain(pending);
            local?;remote?;
            mark!(1);
        }
        check(unsafe{h3vgs_stitch(slot as _,clip as _,3,28,th as _,tw as _,height as _,width as _,shape.y.starts.len() as _,shape.x.starts.len() as _,ys as _,xs as _,yo as _,xo as _,1,program.arena.stream)})?;
        check(unsafe{h3vg_temporal_merge(clip as _,overlap as _,video as _,chunk as _,temporal.chunks as _,height as _,width as _,frames as _,program.arena.stream)})?;
        // This download synchronizes GPU0: the stitch has consumed this chunk's tile buffer
        // before any worker may overwrite it.
        let n=u32::from_le_bytes(program.arena.download(nonfinite,4)?.try_into().unwrap());if n!=0{return Err(format!("nonfinite decoder tile values: {n}").into());}
        mark!(2);
        let next=match &queue{Some(queue)=>{queue.stitched(chunk)?;pending},None=>if chunk+1<temporal.chunks{session.team.dispatch(chunk+1)?}else{0}};
        let emitting=Instant::now();
        let emitted=(||->Result<()>{
        if let Some(output)=frame_stream.as_mut().filter(|_|delivery.is_none()){
            let first=chunk*17;let count=if chunk+1==temporal.chunks{frames-first}else{17.min(frames-first)};
            let count=count.min(output.frames.saturating_sub(first));
            if count>0{
            check(unsafe{h3vg_pt8_range(video as _,pt8 as _,height as _,width as _,frames as _,first as _,count as _,program.arena.stream)})?;
            output.emit(pt8,first,count,&program.arena,height,width)?;
            }
        }
        if let Some((output,head,tail,global_first))=long_output.as_mut(){
            let first=chunk*17;let end=if chunk+1==temporal.chunks{frames}else{(first+17).min(frames)};
            let lo=first.max(*head);let hi=end.min(*tail);
            if lo<hi{
                check(unsafe{h3vg_pt8_range(video as _,pt8 as _,height as _,width as _,frames as _,lo as _,(hi-lo)as _,program.arena.stream)})?;
                output.emit(pt8,*global_first+lo-*head,hi-lo,&program.arena,height,width)?;
            }
        }
        Ok(())})();
        // Never leave dispatched worker replies undrained, even on failure (queued: drained below).
        if let Err(error)=emitted{if queue.is_none(){let _=session.team.drain(next);}return Err(error);}
        emit_max=emit_max.max(emitting.elapsed().as_secs_f64());
        if first_emit<0.0{first_emit=started.elapsed().as_secs_f64();}
        mark!(4);
        pending=next;
        if crate::debug_audit::enabled(){eprintln!("decoded temporal clip {chunk}/{} in {:.3}s",temporal.chunks-1,t.elapsed().as_secs_f64());}
    }
    Ok(())})();
    if let Some(queue)=&queue{
        // Every worker replies once the queue is exhausted or failed: drain them on any outcome.
        if let Err(error)=&decoded{queue.fail(&error.to_string());}
        let replies=session.team.drain(pending);decoded?;let replies=replies?;
        session.team.queued=true;
        session.team.queued_tiles=std::iter::once(own_tiles).chain(replies.iter().map(|r|r["tiles"].as_u64().unwrap_or(0) as usize)).collect();
    }else{decoded?;session.team.queued=false;session.team.queued_tiles.clear();}
    if std::env::var("H3_VAE_CAPTURE_INPUT").as_deref()==Ok("1"){
        fs::write(out.join("decoded-half.raw"),program.arena.download(video,3*frames*pixels*2)?)?;
    }
    if let (Some([_,_,count]),Some(output))=(delivery,frame_stream.as_mut()){
        for first in (0..count).step_by(chunk_frames){
            let n=chunk_frames.min(count-first);
            check(unsafe{h3vg_pt8_range(video as _,pt8 as _,height as _,width as _,frames as _,first as _,n as _,program.arena.stream)})?;
            output.emit(pt8,first,n,&program.arena,height,width)?;
        }
    }
    // After the last chunk: whole-video scan, frame stream completion and tile team cleanup (opt-in split).
    let finishing=Instant::now();
    check(unsafe{h3v_half_nonfinite(video as _,nonfinite as _,(3*frames*pixels) as _,program.arena.stream)})?;
    let nonfinite_count=u32::from_le_bytes(program.arena.download(nonfinite,4)?.try_into().unwrap());
    if nonfinite_count!=0{return Err(format!("nonfinite decoder values before pt8: {nonfinite_count}").into());}
    let scanned=finishing.elapsed().as_secs_f64();
    let (frames_sha,stream_report)=if long_output.is_some(){(Value::Null,json!({"long_slice":true}))}else if let Some(output)=frame_stream.as_mut(){
        let report=output.finish()?;(report["raw_sha256"].clone(),report)
    }else{
        check(unsafe{h3vg_pt8(video as _,pt8 as _,height as _,width as _,frames as _,program.arena.stream)})?;
        let frame_bytes=program.arena.download(pt8,frames*pixels*3)?;fs::write(out.join("frames.raw"),&frame_bytes)?;
        (crate::debug_audit::sha(||hash(&frame_bytes)),Value::Null)
    };
    let streamed=finishing.elapsed().as_secs_f64();
    session.finish()?;
    let finished=finishing.elapsed().as_secs_f64();
    if let Some(dir)=profile_dir.as_ref().filter(|_|long_output.is_none()){
        let _=fs::create_dir_all(dir);
        let record=json!({"schema":"h3-vae-decode-profile-v1","width":width,"height":height,"frames":frames,
            "temporal_chunks":temporal.chunks,"spatial_tiles":[shape.y.starts.len(),shape.x.starts.len()],"active_workers":session.team.active_count(),
            "local_tiles_seconds":phase[0],"remote_tiles_wait_seconds":phase[1],"stitch_merge_check_seconds":phase[2],
            "pt8_seconds":phase[3],"emit_seconds":phase[4],"coordinator_tiles":coordinator_tiles,"wall_seconds":started.elapsed().as_secs_f64(),
            "queued":queued,"queued_tiles_per_gpu":session.team.queued_tiles,"emit_max_seconds":emit_max,"first_emit_seconds":first_emit,
            "finish_split_seconds":{"nonfinite_scan":scanned,"stream_finish":streamed-scanned,"team_finish":finished-streamed},
            "preparation_seconds":preparation_seconds,"emit_split_seconds":stream_report["emit_profile_seconds"],
            "preparation_split_seconds":prep_split,
            "worker_prepare_split_seconds":session.team.report()["workers"].as_array().map(|w|{
                // Slowest worker per phase: the coordinator waits for all of them.
                let mut max=serde_json::Map::new();
                for worker in w{if let Some(split)=worker["prepare_split_seconds"].as_object(){for (k,v) in split{
                    let v=v.as_f64().unwrap_or(0.0);if max.get(k).and_then(Value::as_f64).is_none_or(|m|v>m){max.insert(k.clone(),json!(v));}}}}
                Value::Object(max)}).unwrap_or(Value::Null)});
        let name=format!("decode-{}-{}.json",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d|d.as_nanos()).unwrap_or(0));
        let _=fs::write(dir.join(name),serde_json::to_vec_pretty(&record)?);
        // Per-request copy beside the stage output so the pipeline report can carry it.
        // Streaming decode never creates its output directory; make it for this copy.
        let _=fs::create_dir_all(out);
        let _=fs::write(out.join("decode-profile.json"),serde_json::to_vec_pretty(&record)?);
    }
    let clean=maps(out)?;let report=json!({"passed":clean,"mode":"decode","validation":"execution_only","no_reference_activations_read":true,
        "native_dependencies_clean":if crate::debug_audit::enabled(){json!(clean)}else{Value::Null},"debug_audit":crate::debug_audit::record(),"model":model.manifest,"input_sha256":crate::debug_audit::sha(||hash(&input_bytes)),"frames_sha256":frames_sha,"streaming":stream_report,
        "shape":[frames,height,width,3],"latent_shape":[1,24,temporal.latent_frames,lh,lw],"spatial_tiles":[shape.y.starts.len(),shape.x.starts.len()],"temporal_chunks":temporal.chunks,"padded_latent_frames":temporal.pad_tokens,
        "parallel_vae":session.team.report(),"dtype":"uint8","nonfinite":nonfinite_count,"load_seconds":model_load_seconds,"preparation_seconds":preparation_seconds,"wall_seconds":started.elapsed().as_secs_f64()});
    if crate::debug_audit::enabled(){fs::write(out.join("report.json"),serde_json::to_string_pretty(&report)?+"\n")?;}if !clean{return Err("forbidden dependency".into());}Ok(report)
}

fn postquant(args:&[String])->Result<()> {
    if args.len()!=4{return Err("usage: h3-video-vae postquant AUDIT_ROOT OUTPUT_DIR".into());}
    let root=Path::new(&args[2]);let out=Path::new(&args[3]);fs::create_dir_all(out)?;
    let audit:Value=serde_json::from_slice(&crate::startup_trace::read(root.join("manifest.json"))?)?;let fixture=&audit["fixtures"]["decoder_tile"];
    let mut arena=Arena::new()?;let input=allocate_upload(&mut arena,&read_checked(root,&fixture["pre_quant_input"])?)?;
    let output=arena.alloc(24*7*16*16*2)?;let post=PostQuant::new(&mut arena,root,&audit)?;let gold=read_checked(root,&fixture["input"])?;let mut results=Vec::new();
    for repeat in 0..3{post.forward(&arena,input,output)?;let actual=arena.download(output,gold.len())?;let mut result=compare(&actual,&gold,"torch.float16")?;result["repeat"]=json!(repeat);results.push(result);}
    let clean=maps(out)?;let passed=clean&&results.iter().all(|r|r["passed"]==true);let report=json!({"passed":passed,"mode":"postquant","native_dependencies_clean":clean,"results":results});
    fs::write(out.join("report.json"),serde_json::to_string_pretty(&report)?+"\n")?;println!("{report}");if !passed{return Err("postquant gate failed".into());}Ok(())
}
include!("parallel.rs");
include!("stream.rs");
include!("long_decode.rs");
include!("resident.rs");
pub(super) fn work(args:&[String])->Result<()>{if args[1]=="postquant"{return postquant(args);}let mut resident=Resident::default();let response=resident.execute(args)?;println!("{response}");Ok(())}
