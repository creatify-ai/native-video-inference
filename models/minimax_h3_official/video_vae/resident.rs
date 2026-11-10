// Host lifetime/socket overlay. Numerical bodies remain in full.rs.
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::time::Duration;

unsafe extern "C" {
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> i32;
}

fn device_memory() -> Result<Value> {
    let (mut free, mut total) = (0, 0);
    check(unsafe { cuMemGetInfo_v2(&mut free, &mut total) })?;
    Ok(json!({"free_bytes":free,"total_bytes":total,"used_bytes":total-free,
              "scope":"device-wide; includes other resident component processes"}))
}

struct Context {
    tile_team:Option<TileTeam>,
    request:Option<VideoRequest>,
    model_root: PathBuf,
    shape: Vec<usize>,
    seen_shapes: std::collections::HashSet<Vec<usize>>,
    model: Model,
    // Drop the cuDNN postquant handle before releasing its arena allocations.
    post: Option<PostQuant>,
    program: Program,
    mean: DevicePtr,
    std: DevicePtr,
    persistent_mark: usize,
    persistent_bytes: usize,
    high_water_bytes: usize,
    load_seconds: f64,
    requests_completed: usize,
}

impl Context {
    fn release_idle_workspace(&mut self)->Result<Value>{
        if self.program.arena.allocation_mark()!=self.persistent_mark || self.program.arena.allocation_bytes()!=self.persistent_bytes {
            return Err("cannot release active video VAE workspace".into());
        }
        self.program.arena.sync()?;
        let program=release_conv_workspace(self.program.conv)?;
        let post=if let Some(post)=&self.post{release_conv_workspace(post.ctx)?}else{json!({"released_bytes":0})};
        let team=if let Some(team)=&mut self.tile_team{team.release_idle_workspace()?}else{json!({"released_bytes":0})};
        Ok(json!({"released_bytes":program["released_bytes"].as_u64().unwrap()+post["released_bytes"].as_u64().unwrap()+team["released_bytes"].as_u64().unwrap(),
            "program":program,"post_quant":post,"tile_team":team,"persistent_bytes":self.persistent_bytes,"weights_preserved":true}))
    }
    fn load(root: &Path, stage: &str, shape: Vec<usize>) -> Result<Self> {
        let _trace=crate::startup_trace::Span::new("submodel",stage,0);
        let started = Instant::now();
        let model_root = fs::canonicalize(root)?;
        let model = Model::load(&model_root, stage)?;
        let read_ahead=stage=="encoder" && crate::weight_loader::flag("H3_MODEL_LOAD_ENCODER_READ_AHEAD")?;
        let mut program = Program::load_execution_options(&model.program_root, false, read_ahead)?;
        let mean = allocate_upload(&mut program.arena, &floats(&model.audit["config"]["latents_mean"])?)?;
        let std = allocate_upload(&mut program.arena, &floats(&model.audit["config"]["latents_std"])?)?;
        let post = if stage == "decoder" {
            Some(PostQuant::new(&mut program.arena, &model.audit_root, &model.audit)?)
        } else { None };
        program.arena.sync()?;
        let persistent_mark = program.arena.allocation_mark();
        let persistent_bytes = program.arena.allocation_bytes();
        if std::env::var("H3_VAE_METADATA_PREPARE").as_deref()==Ok("1") {
            let started_metadata=Instant::now();
            // The cuDNN convolution and attention plans of this maximum-tile program. Plans are
            // built only: no kernel is launched and no tensor is read.
            let prepared=(||->Result<()>{
                program.ensure_workspace()?;
                program.prepare_metadata_plans()?;
                if let Some(post)=post.as_ref(){post.prepare_metadata(&program.arena,input_pointer(&program)?)?;}
                Ok(())
            })();
            eprintln!("H3_VIDEO_METADATA stage={} seconds={:.6} passed={}",stage,started_metadata.elapsed().as_secs_f64(),prepared.is_ok());
            if let Err(error)=prepared {eprintln!("video metadata setup deferred: {error}");}
            program.arena.release_after(persistent_mark)?;program.workspace_released();
            if program.arena.allocation_bytes()!=persistent_bytes{return Err("metadata temporary allocations did not return to baseline".into());}
        }
        Ok(Self { tile_team:None, request:None, model_root, seen_shapes:std::collections::HashSet::from([shape.clone()]), shape, model, post, program, mean, std, persistent_mark,
                  persistent_bytes, high_water_bytes:persistent_bytes,
                  load_seconds:started.elapsed().as_secs_f64(), requests_completed:0 })
    }
}

impl Drop for Context {
    fn drop(&mut self) { let _ = self.program.arena.sync(); }
}

#[derive(Default)]
struct Resident {
    encoder: Option<Context>,
    temporal_encoder: Option<TemporalEncoder>,
    decoder: Option<Context>,
    requests_completed: usize,
    input_announced: bool,
}

impl Resident {
    fn release_idle_workspace(&mut self)->Result<Value>{
        let mut components=serde_json::Map::new();let mut released=0u64;
        for (name,ctx) in [("encoder",self.encoder.as_mut()),("decoder",self.decoder.as_mut())]{
            if let Some(ctx)=ctx{let report=ctx.release_idle_workspace()?;released+=report["released_bytes"].as_u64().unwrap();components.insert(name.into(),report);}
        }
        if let Some(ctx)=self.temporal_encoder.as_mut(){
            if ctx.program.arena.allocation_mark()!=ctx.persistent_mark || ctx.program.arena.allocation_bytes()!=ctx.persistent_bytes{return Err("cannot release active temporal encoder workspace".into());}
            ctx.program.arena.sync()?;let report=release_conv_workspace(ctx.program.conv)?;
            released+=report["released_bytes"].as_u64().unwrap();components.insert("reference_encoder".into(),report);
        }
        // SDPA plans are shared by contexts on this thread, unlike conv handles.
        let report=release_dense_workspace()?;released+=report["released_bytes"].as_u64().unwrap();components.insert("thread_sdpa".into(),report);
        Ok(json!({"component":"video_idle_workspace","released_bytes":released,"components":components,"weights_preserved":true,"plans_preserved":true,"pid":std::process::id()}))
    }
    fn execute(&mut self, args: &[String]) -> Result<Value> {
        crate::debug_audit::begin()?;
        let (clean_args,request_context)=request_arguments(args)?;let args=&clean_args;
        let started = Instant::now();
        let mode = args.get(1).ok_or("missing video mode")?.as_str();
        if mode=="release-idle-workspace" {
            if args.len()!=2{return Err("release-idle-workspace takes no arguments".into());}
            return self.release_idle_workspace();
        }
        if ["preload","preload-encoder","preload-output","preload-decoder"].contains(&mode) {
            if ![3,5].contains(&args.len()) || (args.len()==5 && args[4]!="-") {
                return Err("invalid video preload arguments".into());
            }
            let root=Path::new(&args[2]);
            if mode=="preload-decoder" && args.len()==5 && args[3]!="-" {
                return Err("decoder-only preload cannot include a reference encoder".into());
            }
            if mode=="preload-output" {
                let encoder=self.encoder.as_ref().ok_or("output preload requires a loaded encoder")?;
                if encoder.model_root!=fs::canonicalize(root)? || self.decoder.is_some() {
                    return Err("output preload must complete the original unloaded video model".into());
                }
            } else {
                if self.encoder.is_some() || self.decoder.is_some() {
                    return Err("preload requires one unloaded video model".into());
                }
                if mode!="preload-decoder" {self.encoder=Some(Context::load(root,"encoder",vec![256,256])?);}
            }
            if mode!="preload-encoder" {self.decoder=Some(Context::load(root,"decoder",vec![256,256,4])?);}
            if args.len()==5 {
                if args[3]!="-"{self.temporal_encoder=Some(TemporalEncoder::load(Path::new(&args[3]))?);}
            }
            return Ok(json!({"component":"video_vae","model_load_count":1,"requests_completed":self.requests_completed,
                "preload_mode":mode,
                "load_counts":{"encoder":usize::from(self.encoder.is_some()),"decoder":usize::from(self.decoder.is_some()),"reference_encoder":usize::from(self.temporal_encoder.is_some())},"load_seconds":started.elapsed().as_secs_f64()}));
        }
        if mode=="encode-video" {
            if args.len()!=9{return Err("encode-video argument count".into());}
            if self.temporal_encoder.is_none(){self.temporal_encoder=Some(TemporalEncoder::load(Path::new(&args[2]))?);}
            let report=self.temporal_encoder.as_mut().unwrap().execute(args)?;
            self.requests_completed+=1;return Ok(report["resident"].clone());
        }
        let (shape, output_index, stage) = match mode {
            "encode" if args.len()==7 => (vec![768,768],6,"encoder"),
            "encode" if args.len()==9 => (vec![args[7].parse()?,args[8].parse()?],6,"encoder"),
            "decode" if args.len()==5 => (vec![768,768,345],4,"decoder"),
            "decode" if args.len()==8 => (vec![args[5].parse()?,args[6].parse()?,args[7].parse()?],4,"decoder"),
            "decode-long" if args.len()==8 => (vec![args[5].parse()?,args[6].parse()?,args[7].parse()?],4,"decoder"),
            "deliver-rgb" if args.len()==8 => (vec![args[5].parse()?,args[6].parse()?,args[7].parse()?],4,"decoder"),
            _ => return Err("resident video accepts the original encode/decode CLI arguments only".into()),
        };
        let root = fs::canonicalize(&args[2])?;
        let slot = if stage=="encoder" { &mut self.encoder } else { &mut self.decoder };
        let first_load = slot.is_none();
        if first_load { *slot=Some(Context::load(&root,stage,shape.clone())?); }
        let ctx = slot.as_mut().unwrap();
        ctx.request=request_context;
        if ctx.model_root!=root {
            return Err("resident video weights remain bound to the original model root".into());
        }
        if ctx.program.arena.allocation_mark()!=ctx.persistent_mark ||
           ctx.program.arena.allocation_bytes()!=ctx.persistent_bytes {
            return Err("request entered with unreleased temporary allocations".into());
        }
        let shape_cache_hit = ctx.seen_shapes.contains(&shape) && !first_load;
        let shape_changed = ctx.shape!=shape;
        if ctx.seen_shapes.len()>=64 && !shape_cache_hit {ctx.seen_shapes.clear();}
        ctx.seen_shapes.insert(shape.clone());ctx.shape=shape.clone();
        let prepared_shapes=ctx.seen_shapes.len();
        let request_load_seconds = if first_load { ctx.load_seconds } else { 0.0 };
        let memory_before = device_memory()?;
        ctx.program.ensure_workspace()?;
        let outcome = if stage=="encoder" {
            encode_body(args,ctx,started,request_load_seconds)
        } else if mode=="deliver-rgb" {
            deliver_rgb(args,ctx)
        } else {
            decode_body(args,ctx,started,request_load_seconds)
        };
        // No temporary pointer is retained in Context/Program. Reclaim on success
        // and error, after the same stream has completed all queued work.
        let peak_bytes = ctx.program.arena.allocation_bytes();
        let peak_allocations = ctx.program.arena.allocation_mark();
        ctx.high_water_bytes = ctx.high_water_bytes.max(peak_bytes);
        let cleanup = ctx.program.arena.release_after(ctx.persistent_mark);
        cleanup?;
        ctx.program.workspace_released();
        if ctx.program.arena.allocation_mark()!=ctx.persistent_mark ||
           ctx.program.arena.allocation_bytes()!=ctx.persistent_bytes {
            return Err("request temporary GPU allocations did not return to baseline".into());
        }
        let mut report = outcome?;
        ctx.requests_completed += 1;
        let component_requests = ctx.requests_completed;
        let resident_load_seconds = ctx.load_seconds;
        let arena = json!({"persistent_allocations":ctx.persistent_mark,
            "persistent_bytes":ctx.persistent_bytes,"after_allocations":ctx.program.arena.allocation_mark(),
            "after_bytes":ctx.program.arena.allocation_bytes(),"request_peak_allocations":peak_allocations,
            "request_peak_bytes":peak_bytes,"request_temporary_bytes":peak_bytes-ctx.persistent_bytes,
            "high_water_bytes":ctx.high_water_bytes,"temporary_allocations_released":true,
            "sync_before_free":true});
        let spatial_plans=ctx.program.spatial_report();
        let memory_after = device_memory()?;
        let workspace_policy=if ctx.program.workspace_bytes>0{"parameters/plans persistent; tile and outer request buffers synchronized then released"}else{"fixed tile program persistent; outer request buffers synchronized then released"};
        self.requests_completed += 1;
        let details = json!({"component":format!("video_{mode}"),"spatial_plans":spatial_plans,"requests_completed":component_requests,
            "process_requests_completed":self.requests_completed,"model_load_count":1,
            "load_counts":{"encoder":usize::from(self.encoder.is_some()),"decoder":usize::from(self.decoder.is_some())},
            "request_load_seconds":request_load_seconds,"resident_model_load_seconds":resident_load_seconds,
            "request_wall_seconds":started.elapsed().as_secs_f64(),"pid":std::process::id(),
            "memory":{"arena":arena,"device_before":memory_before,"device_after":memory_after},
            "no_reference_activations_read":true,
            "dynamic_shape":{"shape":shape,"shape_changed":shape_changed,"shape_cache_hit":shape_cache_hit,
                "seen_shapes":prepared_shapes,"seen_shapes_limit":64,"shape_prepare_seconds":0.0,"weight_reloaded":false,
                "workspace_policy":workspace_policy},"debug_audit":crate::debug_audit::record()});
        report["resident"] = details.clone();
        fs::write(Path::new(&args[output_index]).join("report.json"),serde_json::to_string_pretty(&report)?+"\n")?;
        if stage=="encoder" && !self.input_announced {
            // Publish only after successful encoding and temporary-buffer cleanup.
            // The host may now queue output preload on this same serial CUDA owner.
            if let Some(path)=std::env::var_os("H3_VIDEO_PRELOAD_INPUT_USED") {fs::write(path,b"ready\n")?;}
            self.input_announced=true;
        }
        Ok(details)
    }
}

struct SocketPath(PathBuf);
impl Drop for SocketPath { fn drop(&mut self) { let _=fs::remove_file(&self.0); } }

pub(super) fn serve(path: &Path) -> Result<()> {
    // Never unlink somebody else's existing socket.
    let listener = UnixListener::bind(path)?;
    let _cleanup = SocketPath(path.to_owned());
    let mut resident = Resident::default();
    eprintln!("{}",json!({"event":"ready","component":"video_vae","pid":std::process::id(),"socket":path}));
    for connection in listener.incoming() {
        let mut connection = connection?;
        connection.set_read_timeout(Some(Duration::from_secs(30)))?;
        connection.set_write_timeout(Some(Duration::from_secs(30)))?;
        let outcome = (|| -> Result<Value> {
            let mut line = String::new();
            let reader = BufReader::new(connection.try_clone()?);
            reader.take(65537).read_line(&mut line)?;
            if line.len()>65536 || !line.ends_with('\n') { return Err("expected one bounded JSON line".into()); }
            let request: Value = serde_json::from_str(&line)?;
            let supplied = request["args"].as_array().ok_or("request.args must be an array")?;
            let mut args = vec!["h3-video-vae".to_owned()];
            for arg in supplied { args.push(arg.as_str().ok_or("arguments must be strings")?.to_owned()); }
            resident.execute(&args)
        })();
        let response = match outcome {
            Ok(details) => json!({"ok":true,"error":Value::Null,"details":details}),
            Err(error) => json!({"ok":false,"error":error.to_string(),"details":{"component":"video_vae","pid":std::process::id()}}),
        };
        serde_json::to_writer(&mut connection,&response)?;
        connection.write_all(b"\n")?;
        connection.flush()?;
    }
    Ok(())
}

impl PostQuant {
    fn prepare_metadata(&self,arena:&Arena,output:DevicePtr)->Result<()> {
        let(h,w)=(16i64,16i64);
        let spec=ConvSpec{ndim:5,dtype:1,allow_tf32:1,input_dims:[1,24,7,h,w],input_strides:[24*7*h*w,7*h*w,h*w,w,1],
            weight_dims:[24,24,1,1,1],weight_strides:[24,1,1,1,1],output_dims:[1,24,7,h,w],output_strides:[24*7*h*w,7*h*w,h*w,w,1],
            stride:[1,1,1],dilation:[1,1,1],..Default::default()};
        let code=unsafe{h3_conv_prepare(self.ctx,&spec,self.input as _,self.w as _,output as _,arena.stream)};
        if code==0{Ok(())}else{Err(unsafe{CStr::from_ptr(h3_conv_error())}.to_string_lossy().into_owned().into())}
    }
}
