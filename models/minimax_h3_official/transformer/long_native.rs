// The same maintained DiT forward executes each legal window. Global samples,
// overlap accumulators and rolling x0 anchors stay on the GPU for the request.
#[derive(Clone)]
struct PreparedLong {
    manifest:Value,
    shape:h3_native_pipeline::Shape,
    windows:Vec<PreparedRequest>,
    video:Vec<u8>,audio:Vec<u8>,
}
impl PreparedLong {
    fn read(root:&Path,manifest:Value)->Result<PreparedRequest> {
        if manifest["status"]!="complete" || manifest["producer"]!="h3-native-pipeline" ||
           manifest["workflow"]!="fl2va" || crate::recipe::Recipe::from_request(&manifest).is_err() || manifest["gold_activations_used"]!=false ||
           manifest.get("cache_transaction").is_some() || manifest.get("denoise_phase").is_some() {
            return Err("invalid native FL2VA long request".into());
        }
        let s=&manifest["shape"];
        let integer=|name:&str|->Result<usize>{Ok(usize::try_from(s[name].as_u64().ok_or("long geometry missing")?)?)};
        let shape=h3_native_pipeline::Shape::long(integer("width")?,integer("height")?,integer("num_frames")?,integer("keyframe_count")?)?;
        if s!=&shape.json(){return Err("long request shape metadata differs".into());}
        let plan=h3_native_pipeline::long_video::Plan::new(shape.num_frames,shape.num_frames as f64/24.0,true)?;
        if manifest["plan"]!=serde_json::to_value(&plan)?{return Err("long request plan differs from the reference geometry".into());}
        let read=|name:&str,rows:usize,cols:usize|->Result<Vec<u8>>{
            let spec=TensorSpec::parse(&manifest["inputs"][name])?;
            if spec.dtype!="torch.float32" || spec.shape!=[1,rows,cols]{return Err("long input shape/dtype differs".into());}
            let bytes=spec.read(root)?;
            if bytes.chunks_exact(4).any(|b|!f32::from_le_bytes(b.try_into().unwrap()).is_finite()){
                return Err("nonfinite long input".into());
            }Ok(bytes)
        };
        let video=read("hidden_states",shape.video_rows(),VIDEO)?;
        let audio=read("audio_hidden_states",shape.audio_rows(),AUDIO)?;
        let entries=manifest["windows"].as_array().ok_or("long windows missing")?;
        if entries.len()!=plan.windows.len(){return Err("long window count differs".into());}
        let mut windows=Vec::new();let frame_bytes=shape.rows_per_frame()*VIDEO*4;
        for (i,(entry,w)) in entries.iter().zip(&plan.windows).enumerate(){
            let anchors=w.anchors(shape.keyframe_count>0,shape.keyframe_count==2,entries.len()>1);
            let next=plan.windows.get(i+1);
            // JSON parses numbers through f64. Compare their declared FP32
            // values, avoiding a spurious f64 decimal round-trip difference.
            let video_weights:Vec<f32>=serde_json::from_value(entry["video_weights"].clone())?;
            let audio_weights:Vec<f32>=serde_json::from_value(entry["audio_weights"].clone())?;
            if entry["geometry"]!=serde_json::to_value(w)? || entry["anchors"]!=anchors ||
                entry["request"]!=format!("windows/{i:03}") ||
                video_weights!=h3_native_pipeline::long_video::fuse_weights(w.latent_frames[1]-w.latent_frames[0],w.overlap_latent_frames,next.map_or(0,|n|n.overlap_latent_frames))? ||
                audio_weights!=h3_native_pipeline::long_video::fuse_weights(w.audio_latents[1]-w.audio_latents[0],w.overlap_audio_latents,next.map_or(0,|n|n.overlap_audio_latents))? {
                return Err("long window anchors/weights/geometry differ".into());
            }
            let input=root.join(format!("windows/{i:03}"));
            // Reject nested long descriptors before recursing into the reader.
            let local:Value=serde_json::from_slice(&crate::startup_trace::read(input.join("manifest.json"))?)?;
            if local["schema"]!="h3-native-request-v1" || local.get("cache_transaction").is_some() ||
               local.get("denoise_phase").is_some(){return Err("long window must be an ordinary native layout".into());}
            let p=PreparedRequest::read(&input)?;
            let count=usize::from(!anchors["first"].is_null())+usize::from(!anchors["last"].is_null());
            let expected=h3_native_pipeline::Shape::new(shape.width,shape.height,w.pixel_frames)?.with_keyframes(count)?;
            if p.manifest["shape"]!=expected.json() || p.workflow()?!="fl2va" || p.nfe()?!=crate::recipe::Recipe::from_request(&manifest)?.nfe {return Err("long window input shape differs".into());}
            let mut v=Vec::new();
            if !anchors["first"].is_null(){v.extend_from_slice(&video[..frame_bytes]);}
            if !anchors["last"].is_null(){let start=if anchors["last"]=="real"{(shape.keyframe_count-1)*frame_bytes}else{0};
                v.extend_from_slice(&video[start..start+frame_bytes]);}
            v.extend_from_slice(&video[(shape.keyframe_count+w.latent_frames[0])*frame_bytes..(shape.keyframe_count+w.latent_frames[1])*frame_bytes]);
            let mut a=Vec::new();
            for c in 0..2{a.extend_from_slice(&audio[(c*shape.num_audio_latents+w.audio_latents[0])*AUDIO*4..(c*shape.num_audio_latents+w.audio_latents[1])*AUDIO*4]);}
            if p.data["hidden_states"]!=v || p.data["audio_hidden_states"]!=a {return Err("long window samples are not slices of the global noise draw".into());}
            if let Some(first)=windows.first(){let first:&PreparedRequest=first;
                if first.data["encoder_hidden_states"]!=p.data["encoder_hidden_states"]{return Err("long text conditioning must be shared".into());}}
            windows.push(p);
        }
        let mut first=windows.first().ok_or("empty long plan")?.clone();
        first.long=Some(Box::new(Self{manifest,shape,windows,video,audio}));Ok(first)
    }
}

unsafe extern "C" {
    fn h3_long_anchor(out:*mut c_void,clean:*const c_void,noise:*const c_void,n:i64,t:f32,stream:Stream)->c_int;
    fn h3_long_color_workspace(rows:i64,cols:c_int)->usize;
    fn h3_long_color_shift(previous:*const c_void,sample:*const c_void,pred:*const c_void,scratch:*mut c_void,bytes:usize,
        out:*mut c_void,rows:i64,cols:c_int,sigma:f32,stream:Stream)->c_int;
    fn h3_long_accumulate(sample:*const c_void,pred:*const c_void,color:*const c_void,weights:*const c_void,
        sum:*mut c_void,weight_sum:*mut c_void,x0:*mut c_void,rows:i64,cols:c_int,rows_per_weight:c_int,sigma:f32,stream:Stream)->c_int;
    fn h3_long_blend(sample:*const c_void,velocity:*mut c_void,weights:*const c_void,x0:*mut c_void,
        rows:i64,cols:c_int,sigma:f32,stream:Stream)->c_int;
}

impl Runtime {
    fn execute_long(&mut self,request:PreparedLong)->Result<Value>{
        let output=self.output.clone();let started=Instant::now();let shape=request.shape;
        let override_value=match std::env::var("H3_NATIVE_LONG_EVALUATIONS_FILE") {
            Ok(path)=>Ok(fs::read_to_string(path)?.trim().to_owned()),
            Err(std::env::VarError::NotPresent)=>std::env::var("H3_NATIVE_LONG_EVALUATIONS"),
            Err(error)=>return Err(error.into()),
        };
        let evaluations=match override_value {
            Ok(value) if crate::debug_audit::enabled()=>value.parse::<usize>()?,
            Ok(_)=>return Err("long evaluation override requires numerical diagnostic mode".into()),
            Err(std::env::VarError::NotPresent)=>self.nfe(),Err(error)=>return Err(error.into()),
        };
        if !(1..=self.nfe()).contains(&evaluations){return Err("long diagnostic evaluations exceed the selected recipe".into());}
        let plan=h3_native_pipeline::long_video::Plan::new(shape.num_frames,shape.num_frames as f64/24.0,true)?;
        let entries=request.manifest["windows"].as_array().ok_or("long window metadata")?;
        let rows=shape.rows_per_frame();let nv=shape.num_latent_frames*rows;let na=shape.audio_rows();
        let prefix=shape.condition_rows()*VIDEO*4;
        let mut global=Arena::new()?;
        let video=global.upload(&request.video[prefix..])?;let audio=global.upload(&request.audio)?;
        let vv=global.alloc(video.bytes)?;let av=global.alloc(audio.bytes)?;
        let vw=global.alloc(nv*4)?;let aw=global.alloc(na*4)?;
        let current=global.alloc(video.bytes)?;let previous=global.alloc(video.bytes)?;
        let color=global.alloc(VIDEO*4)?;
        let max_overlap=plan.windows.iter().map(|w|w.overlap_latent_frames*rows).max().unwrap_or(0);
        let color_scratch=global.alloc(unsafe{h3_long_color_workspace(max_overlap.max(1) as i64,VIDEO as c_int)})?;
        let mut weights=Vec::new();let mut noise=Vec::new();
        for (i,entry) in entries.iter().enumerate(){
            let v:Vec<f32>=serde_json::from_value(entry["video_weights"].clone())?;
            let a:Vec<f32>=serde_json::from_value(entry["audio_weights"].clone())?;
            weights.push((global.upload(&h3_native_pipeline::f32_bytes(&v))?,global.upload(&h3_native_pipeline::f32_bytes(&a))?));
            let mut fixed=Vec::new();
            for side in 0..2 {let mut rng=h3_native_pipeline::CpuGenerator::new(90000+2*i as u64+side);
                fixed.push(global.upload(&h3_native_pipeline::f32_bytes(&rng.normal(rows*VIDEO)?))?);}
            noise.push(fixed);
        }
        let mut phase=crate::request_transaction::Phase{id:crate::integrity::digest(output.to_string_lossy().as_bytes()),index:0,count:2};
        if let Some(sharded)=&self.resident_model.sharded{sharded.begin_request(Some(&phase))?;}
        // The reference planner yields one full-window geometry and at most
        // one shorter tail. Keep that tail only for this request, without a
        // weight reload or a per-window host round-trip of the global samples.
        let mut tail:Option<Runtime>=None;let mut rebuilt=0usize;let mut sparse_calls=0u64;
        let mut results=Vec::new();global.sync()?;let denoise_started=Instant::now();
        for step in 0..evaluations {
            for buffer in [vv,av,vw,aw]{cuda(unsafe{cudaMemsetAsync(buffer.ptr,0,buffer.bytes,global.stream)})?;}
            global.sync()?;let mut covered=vec![false;shape.num_latent_frames];
            for (index,window) in plan.windows.iter().enumerate(){
                let start=Instant::now();let prepared=&request.windows[index];
                let target=output.join(format!("windows/{index:03}"));fs::create_dir_all(&target)?;
                let runtime=if self.prepared_fits(prepared)?{&mut *self}else{
                    let fits=tail.as_ref().map(|r|r.prepared_fits(prepared)).transpose()?.unwrap_or(false);
                    if !fits{
                        drop(tail.take());
                        tail=Some(Runtime::new_shared(&self.root,&self.router_root,&self.dense_path,&target,&prepared.root,Some(self.resident_model.clone()))?);rebuilt+=1;
                    }tail.as_mut().ok_or("long tail layout missing")?
                };
                runtime.output=target;
                if let Some(cp)=&runtime.cp{cp.group.bind(runtime.arena.stream)?;}
                if !runtime.scratch.contains_key("video_embeds"){
                    runtime.scratch.insert("video_embeds",runtime.arena.alloc(runtime.nv*HIDDEN*2)?);
                    runtime.scratch.insert("audio_embeds",runtime.arena.alloc(runtime.na*HIDDEN*2)?);
                }
                if runtime.condition_video>0{
                    runtime.arena.write(runtime.latent_video.prefix(runtime.condition_video*VIDEO*4)?,&prepared.data["hidden_states"][..runtime.condition_video*VIDEO*4])?;
                }
                runtime.arena.write(runtime.conditioning,&prepared.data["encoder_hidden_states"])?;
                let video_rows=(window.latent_frames[1]-window.latent_frames[0])*rows;
                let local_video=runtime.latent_video.slice(runtime.condition_video*VIDEO*4,video_rows*VIDEO*4)?;
                let global_video=video.slice(window.latent_frames[0]*rows*VIDEO*4,local_video.bytes)?;
                runtime.arena.copy(local_video,global_video)?;
                let audio_length=window.audio_latents[1]-window.audio_latents[0];
                for channel in 0..2{
                    runtime.arena.copy(runtime.latent_audio.slice(channel*audio_length*AUDIO*4,audio_length*AUDIO*4)?,
                        audio.slice((channel*shape.num_audio_latents+window.audio_latents[0])*AUDIO*4,audio_length*AUDIO*4)?)?;
                }
                for (side,name,frame) in [(0,"first",window.latent_frames[0]),(1,"last",window.latent_frames[1]-1)]{
                    if entries[index]["anchors"][name]!="carried"{continue;}
                    let fresh=side==0 && covered[frame];if !fresh && step==0{continue;}
                    let source=if fresh{current}else{previous}.slice(frame*rows*VIDEO*4,rows*VIDEO*4)?;
                    let target=runtime.latent_video.slice(side*rows*VIDEO*4,rows*VIDEO*4)?;
                    cuda(unsafe{h3_long_anchor(target.ptr,source.ptr,noise[index][side].ptr,(rows*VIDEO)as i64,0.999,runtime.arena.stream)})?;
                }
                profile::set_step(step as i32);runtime.forward(step,None)?;
                let schedule=runtime.steps[step];let sigma=1.0f32-schedule.video_t;
                let pred=runtime.s("velocity_video").slice(runtime.condition_video*VIDEO*4,local_video.bytes)?;
                let shift=if index>0 && window.overlap_latent_frames>0{
                    let overlap=window.overlap_latent_frames*rows;
                    let earlier=current.slice(window.latent_frames[0]*rows*VIDEO*4,overlap*VIDEO*4)?;
                    cuda(unsafe{h3_long_color_shift(earlier.ptr,global_video.ptr,pred.ptr,color_scratch.ptr,color_scratch.bytes,color.ptr,
                        overlap as i64,VIDEO as c_int,sigma,runtime.arena.stream)})?;color.ptr
                }else{ptr::null_mut()};
                let start_row=window.latent_frames[0]*rows;
                cuda(unsafe{h3_long_accumulate(global_video.ptr,pred.ptr,shift,weights[index].0.ptr,
                    vv.slice(start_row*VIDEO*4,local_video.bytes)?.ptr,vw.slice(start_row*4,video_rows*4)?.ptr,
                    current.slice(start_row*VIDEO*4,local_video.bytes)?.ptr,video_rows as i64,VIDEO as c_int,rows as c_int,sigma,runtime.arena.stream)})?;
                covered[window.latent_frames[0]..window.latent_frames[1]].fill(true);
                for channel in 0..2{
                    let offset=channel*shape.num_audio_latents+window.audio_latents[0];
                    let pred=runtime.s("velocity_audio").slice(channel*audio_length*AUDIO*4,audio_length*AUDIO*4)?;
                    cuda(unsafe{h3_long_accumulate(ptr::null(),pred.ptr,ptr::null(),weights[index].1.ptr,
                        av.slice(offset*AUDIO*4,pred.bytes)?.ptr,aw.slice(offset*4,audio_length*4)?.ptr,ptr::null_mut(),
                        audio_length as i64,AUDIO as c_int,1,1.0f32-schedule.audio_t,runtime.arena.stream)})?;
                }
                runtime.arena.sync()?;
                sparse_calls+=runtime.execution.sparse_layers(step,runtime.seq)? as u64
                    *runtime.router.as_ref().map_or(1,|r|HEADS/r.plan.heads)as u64;
                if crate::debug_audit::enabled(){
                    runtime.write(&format!("steps/{step:02}/video_velocity.raw"),runtime.s("velocity_video"))?;
                    runtime.write(&format!("steps/{step:02}/audio_velocity.raw"),runtime.s("velocity_audio"))?;
                    runtime.write(&format!("steps/{step:02}/input_video.raw"),runtime.latent_video)?;
                    runtime.write(&format!("steps/{step:02}/input_audio.raw"),runtime.latent_audio)?;
                }
                let event=json!({"event":"native.long_video.window_completed","step":step,"window":index,
                    "sequence":runtime.seq,"sparse":runtime.execution.sparse(step,runtime.seq)?,"wall_seconds":start.elapsed().as_secs_f64()});
                if runtime.cp.as_ref().map_or(true,|cp|cp.group.rank==0){eprintln!("{event}");}
                results.push(event);
            }
            let s=self.steps[step];
            cuda(unsafe{h3_long_blend(video.ptr,vv.ptr,vw.ptr,previous.ptr,nv as i64,VIDEO as c_int,1.0-s.video_t,global.stream)})?;
            cuda(unsafe{h3_long_blend(ptr::null(),av.ptr,aw.ptr,ptr::null_mut(),na as i64,AUDIO as c_int,1.0-s.audio_t,global.stream)})?;
            for(sample,velocity,t,ratio)in[(video,vv,s.video_t,s.video_ratio),(audio,av,s.audio_t,s.audio_ratio)]{
                cuda(unsafe{h3_euler_f32(sample.ptr.cast(),velocity.ptr.cast(),sample.ptr.cast(),(sample.bytes/4)as i64,1.0-t,ratio,global.stream)})?;
            }
            global.sync()?;
            if crate::debug_audit::enabled(){let path=output.join(format!("steps/{step:02}"));fs::create_dir_all(&path)?;
                fs::write(path.join("video_latents.raw"),global.download(video)?)?;fs::write(path.join("audio_latents.raw"),global.download(audio)?)?;}
            profile::collect_step()?;
        }
        global.sync()?;let denoise_seconds=denoise_started.elapsed().as_secs_f64();self.output=output.clone();
        if let Some(cp)=&self.cp{cp.group.bind(self.arena.stream)?;}
        if let Some(sharded)=&self.resident_model.sharded{
            sharded.drain(self.arena.stream)?;phase.index=1;sharded.begin_request(Some(&phase))?;sharded.finish_request(self.arena.stream)?;
        }
        let mut final_video=request.video[..prefix].to_vec();final_video.extend(global.download(video)?);
        let final_audio=global.download(audio)?;
        for bytes in [&final_video,&final_audio]{if finite_f32(bytes)?["passed"]!=true{return Err("long final sample nonfinite".into());}}
        fs::write(output.join("final_video.raw"),final_video)?;fs::write(output.join("final_audio.raw"),final_audio)?;
        self.long_sparse_calls=Some(sparse_calls);
        Ok(json!({"passed":true,"mode":"generate","long_video":true,"steps_completed":evaluations,"windows":plan.windows.len(),
            "plan":plan,"window_layout_builds":rebuilt,"results":results,"execution":self.execution.json(),
            "denoise_seconds":denoise_seconds,"wall_seconds":started.elapsed().as_secs_f64(),"global_samples_device_resident":true,
            "debug_audit":crate::debug_audit::record(),"shape":{"sequence":self.seq,"video":self.nv,"audio":self.na,"text":self.nt},
            "long_video_report":{"mode":"fused","num_windows":plan.windows.len(),"text_encodings":1,
                "windows":request.manifest["windows"],"frozen_audio":false,"repair":Value::Null,
                "global_sample_steps":evaluations,"forward_evaluations":evaluations*plan.windows.len()},
            "global_scratch_bytes":global.allocations.iter().map(|b|b.bytes).sum::<usize>(),"request":request.manifest,
            "numerics":self.numerics.name(),"reference_validation_enabled":false,"ordinary_generation":true}))
    }
}
