// Ref2VA uses the same resident ownership, plan cache and numerical methods.
impl ConditionerServer {
    fn request_ref2va(&mut self,args:&[String])->Result<Value> {
        crate::debug_audit::begin()?;self.in_gpu=false;
        let started=Instant::now();
        if args.len()!=5{return Err("infer-ref2va MODEL PROMPT NORMALIZED_DESCRIPTOR OUTPUT".into());}
        let root=fs::canonicalize(&args[1])?;let output=Path::new(&args[4]);
        if !output.is_absolute() || output.exists(){return Err("fresh absolute conditioner output required".into());}
        let (index,manifest_sha)=crate::dynamic_request::checked_model(&root)?;
        let fixed=json!({"model_root":root,"model_manifest_sha256":manifest_sha});
        if self.fixed.as_ref().is_some_and(|v|v!=&fixed){return Err("resident conditioner model identity changed".into());}
        let prepared=output.join("prepared");
        let (mut prep,image_pixels,video_pixels)=crate::ref2va::plan(&root,Path::new(&args[2]),Path::new(&args[3]),&prepared)?;
        let images:Vec<(usize,usize)>=serde_json::from_value(prep["grids"].clone())?;
        let videos:Vec<(usize,usize)>=serde_json::from_value(prep.get("video_grids").cloned().unwrap_or(json!([])))?;
        let &(h,w)=images.iter().chain(&videos).max_by_key(|&&(h,w)|h*w).ok_or("missing vision grids")?;
        let text_rows=prep["token_count"].as_u64().ok_or("text rows")? as usize;
        // A parallel rank prepares only the pixel rows it embeds; one process, or
        // rank 0 alone below the parallel length, writes the whole tensors.
        let role=self.group.as_ref().map_or(Role::Single,|g|g.role(text_rows));
        let local_pixels=match (role,self.group.as_ref()) {
            (Role::Parallel,Some(group))=>Some((image_pixels.rows(vision_slice(&images,group.world,group.rank))?,
                video_pixels.rows(vision_slice(&videos,group.world,group.rank))?)),
            (Role::Single,_)=>{crate::ref2va::write(&prepared,&mut prep,&image_pixels,&video_pixels)?;None}
            _=>None,
        };
        let scratch=prep["scratch_estimated_bytes"].as_u64().ok_or("scratch estimate")? as usize;
        fs::copy(root.join("model_config/config.json"),output.join("model-config.json"))?;
        let startup_preparation=self.adopt_startup_backend()?;
        let (mut free,mut total)=(0usize,0usize);cuda(unsafe{cudaMemGetInfo(&mut free,&mut total)})?;
        let weights=if self.model.is_none(){self.initial_weight_bytes(&index)?}else{0};
        let reserve=crate::dynamic_request::budget("H3_QWEN_VENDOR_RESERVE_BYTES",2usize<<30)?;
        // Sharded ranks load together even when only rank 0 computes this request;
        // an idle rank that loads needs room for its weights, not for scratch.
        let joins_load=self.text_shards.is_some() && self.model.is_none();
        let required=crate::dynamic_request::device_memory_required(role!=Role::Idle||joins_load,
            if role==Role::Idle{0}else{scratch},weights,reserve)?;
        let admitted=required<=free;
        if !admitted {eprintln!("conditioner memory admission rejected: rank={} role={role:?} free_bytes={free} required_bytes={required} scratch_bytes={scratch} weights_bytes={weights} reserve_bytes={reserve}",self.group.as_ref().map_or(0,|g|g.rank));}
        self.admit_memory(admitted,"insufficient conditioner memory for reference layout")?;
        if role==Role::Idle && !joins_load {return self.idle_reply(started,text_rows);}
        let parallel=role==Role::Parallel;
        let world=self.group.as_ref().map_or(1,|g|g.world);
        let decision=if role==Role::Idle{None}else{Some(self.cache.admit_references_for_world(&images,&videos,text_rows,if parallel{world}else{1})?)};
        let layout=Layout{grid_h:h,grid_w:w,text_rows,keyframe_count:1};
        let preprocess_seconds=started.elapsed().as_secs_f64();
        self.in_gpu=true;unsafe{h3_linear_fallback_reset();}
        let mut load_seconds=0.0;
        if self.model.is_none(){
            // A rank group loads weights only; reconfigure then builds its head-group plans.
            let loading=Instant::now();let mut model=Model::load(&root,None,if role==Role::Single{layout}else{Layout::WEIGHTS_ONLY},self.text_shards)?;model.arena.sync()?;
            load_seconds=loading.elapsed().as_secs_f64();self.initial_load=load_seconds;
            self.weights_end=model.parameter_boundary()?;
            self.attach(&mut model)?;
            self.identity=server_identity(&model)?;self.model=Some(model);self.fixed=Some(fixed);
        }
        if role==Role::Idle {self.in_gpu=false;return self.idle_reply(started,text_rows);}
        let decision=decision.ok_or("conditioner plan admission missing")?;
        let model=self.model.as_mut().ok_or("resident model")?;
        let (changed,text_pending)=reconfigure(model,layout,self.weights_end,decision.clear,parallel,parallel.then_some(world))?;
        if text_pending{attach_parallel_text_workspace(model,self.weights_end,world)?;}
        if changed{self.plan_generation+=1;}
        let persistent=model.arena.allocations.len();
        if persistent!=self.weights_end+2{return Err("unreleased conditioner scratch".into());}
        // Variable grids can grow the vision workspace inside vision_grids_file.
        // Save the persistent pointers; any replacement belongs to this request.
        let workspaces=model.attention_workspace;
        let before=server_memory(model)?;
        let forward=Instant::now();
        let mut phases=Phases::new(model.arena.stream);phases.mark()?;
        // The first decoder blocks copy while the vision towers run.
        if let Some(text)=&mut model.text {text.prefetch()?;}
        let plan=match (parallel,self.group.as_ref()) {(true,Some(group))=>Some(model.plan_text_parallel(group,&prepared)?),_=>None};
        let vision=match (images.is_empty(),parallel,self.group.as_mut()) {
            (true,_,_)=>None,
            (false,true,Some(group))=>Some(model.vision_parallel(group,&images,&local_pixels.as_ref().ok_or("rank pixel rows missing")?.0)?),
            _=>Some(model.vision_grids(&prepared,&images)?),
        };
        let video=match (videos.is_empty(),parallel,self.group.as_mut()) {
            (true,_,_)=>None,
            (false,true,Some(group))=>Some(model.vision_parallel(group,&videos,&local_pixels.as_ref().ok_or("rank pixel rows missing")?.1)?),
            _=>Some(model.vision_grids_file(&prepared,&videos,"pixel_values_videos.raw")?),
        };
        phases.mark()?;
        let hidden=match (plan,self.group.as_mut()) {
            (Some(plan),Some(group))=>model.text_parallel(group,plan,vision,video)?,
            _=>model.text_multimodal(&prepared,vision,video)?,
        };
        phases.mark()?;
        model.arena.sync()?;let forward_seconds=forward.elapsed().as_secs_f64();
        let gpu_phases=phases.seconds()?;drop(phases);
        let profile=match self.group.as_mut(){Some(group)=>group.take_profile()?,None=>Value::Null};
        // Every rank holds the gathered rows; rank 0 alone delivers them.
        let leader=self.group.as_ref().is_none_or(|g|g.rank==0);
        let after=server_memory(model)?;let conditioning=if leader{model.arena.read(hidden)?}else{vec![]};
        if conditioning.chunks_exact(2).any(|b|!f32::from_bits((u16::from_le_bytes([b[0],b[1]]) as u32)<<16).is_finite()) {
            return Err("nonfinite Ref2VA conditioning".into());
        }
        if leader {
            fs::write(output.join("conditioning.raw"),&conditioning)?;
            fs::copy(prepared.join("text_token_tags.raw"),output.join("text_token_tags.raw"))?;
        }
        let (freed_count,freed_bytes)=release_after(model,persistent)?;
        model.attention_workspace=workspaces;
        if server_identity(model)?!=self.identity{return Err("resident weight/handle identity changed".into());}
        let attention_entries=unsafe{q3_attention_cache_size()}+unsafe{q3_cp_attention_cache_size()};let conv_entries=unsafe{q3_conv_cache_size(model.conv)};
        if attention_entries+conv_entries!=self.cache.entries.len(){return Err("reference operator plan cache mismatch".into());}
        self.requests+=1;
        let details=json!({"component":"conditioner","workflow":"ref2va","requests_completed":self.requests,
            "model_load_count":1,"request_load_seconds":load_seconds,"initial_model_load_seconds":self.initial_load,
            "weights_and_handles_identity_sha256":self.identity,"preprocess_seconds":preprocess_seconds,
            "forward_seconds":forward_seconds,"stream_phase_seconds":{"vision":gpu_phases[0],"text":gpu_phases[1]},"profile":profile,
            "startup_backend_preparation":startup_preparation,
            "temporary_allocations_freed":freed_count,"temporary_bytes_freed":freed_bytes,
            "memory_before_forward":before,"memory_after_forward":after,"memory":server_memory(model)?,
            "scratch_released":true,"plan_cache":self.cache.record(&decision),"image_grids":images,"video_grids":videos,
            "parallel":self.group.as_ref().map(|g|g.report(role,text_rows)),"text_shards":model.text.as_ref().map(|t|t.report())});
        let report=json!({"schema":"h3-native-conditioner-v1","passed":true,"workflow":"ref2va","delivered":leader,
            "model_manifest_sha256":manifest_sha,"gold_activations_read_for_computation":false,"validation_mode":false,
            "conditioning":{"shape":[1,text_rows,5120],"bytes":conditioning.len(),"nonfinite":0,
                "sha256":crate::debug_audit::sha(||hash(&conditioning))},"preparation":prep,
            "wall_seconds":started.elapsed().as_secs_f64(),"resident":details,"debug_audit":crate::debug_audit::record()});
        fs::write(output.join("report.json"),serde_json::to_vec_pretty(&report)?)?;
        self.in_gpu=false;Ok(details)
    }
}
