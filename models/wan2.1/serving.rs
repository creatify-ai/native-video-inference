//! Standard resident ABI for official Wan. Core owns immutable reads/copies;
//! the actor owns component views and the final-DiT retirement boundary.
use crate::{clip::ClipVisual,cuda::{Buffer,Device},dit::Dit,frontend::{self,Request},gold::{Gold,sha256_hex},
    pipeline,preload_contract::{self,COMPONENTS},t5::T5,vae::Vae,weights::DeviceWeights,Result};
use native_core_infra::{backend_service::{Context,Session as ResidentSession},checkpoint::Source,
    model_load::{self,HostWeights,Loaded,Options,Verify},native_backend::Pin};
use serde::Deserialize;
use serde_json::{json,Value};
use std::{collections::BTreeMap,path::Path,rc::Rc,sync::Arc,time::Instant};

#[derive(Deserialize)]#[serde(deny_unknown_fields)]
struct Profile{schema:String,checkpoint:Source,tokenizer:Pin,geometry:[usize;3]}
impl Profile{
    fn read(path:&Path)->Result<Self>{
        let value:Self=serde_json::from_slice(&std::fs::read(path)?)?;
        if value.schema!="wan21-resident-profile-v1" || value.geometry.iter().any(|n|*n==0)
            || value.geometry[0]%16!=0 || value.geometry[1]%16!=0 || value.geometry[2]%4!=1{
            return Err("Wan profile geometry differs".into());
        }
        value.tokenizer.verify()?;Ok(value)
    }
}
pub struct Session{
    // The Box address is stable. Component views are always dropped before
    // their Loaded owner, and all owners before this context/stream.
    t5:Option<T5<'static>>,clip:Option<ClipVisual<'static>>,vae:Option<Vae<'static>>,pub(crate) dit:Option<Dit<'static>>,
    pub(crate) gpu:BTreeMap<String,Rc<Loaded>>,host:BTreeMap<String,Arc<HostWeights>>,
    source:Option<Source>,profile_sha:Option<String>,generation:u64,observer:Option<Gold>,
    peer:Option<crate::cfg_parallel::CfgWorker>,pub(crate) device:Box<Device>,
    /// The last request's T5/CLIP/VAE conditioning and its key: a request with the same prompts,
    /// image, geometry and profile reuses it (the same deterministic results) instead of recomputing.
    conditioning:Option<(String,pipeline::Conditioning)>,
}
impl Session{
    pub fn open(options:Value)->Result<Self>{
        let world=options["world_size"].as_u64().ok_or("Wan world size missing")?;
        if !matches!(world,1|2){return Err("official Wan requires one or two physical GPUs".into());}
        let mut session=Self::open_rank(0)?;
        if world==2{session.peer=Some(crate::cfg_parallel::CfgWorker::open()?);}
        Ok(session)
    }
    pub(crate) fn open_rank(rank:i32)->Result<Self>{
        Ok(Self{t5:None,clip:None,vae:None,dit:None,gpu:BTreeMap::new(),host:BTreeMap::new(),
            source:None,profile_sha:None,generation:0,observer:None,peer:None,device:Box::new(Device::open_index(rank)?),
            conditioning:None})
    }
    pub(crate) fn uuid(&self)->Result<String>{Ok(native_core_infra::device_context::device_uuid(self.device.index)?)}
    fn components(&self)->&'static [&'static str]{if self.device.index==1{&["dit"]}else{&COMPONENTS}}
    fn world(&self)->usize{if self.peer.is_some(){2}else{1}}
    fn combine_ranks(&self,mut primary:Value,peer:Option<Value>)->Result<Value>{
        if let Some(peer)=peer{
            let ranks=vec![primary.clone(),peer];
            primary["host_pinned_bytes"]=json!(ranks.iter().map(|r|r["host_pinned_bytes"].as_u64().unwrap_or(0)).sum::<u64>());
            primary["device_bytes_uploaded"]=json!(ranks.iter().map(|r|r["device_bytes_uploaded"].as_u64().unwrap_or(0)).sum::<u64>());
            primary["rank_receipts"]=json!(ranks);primary["all_ranks_completed"]=json!(true);primary["world_size"]=json!(2);
        }Ok(primary)
    }
    fn admit_source(&self,profile:&Profile)->Result<native_core_infra::checkpoint::Declaration>{
        if let Some(old)=&self.source{if old.sha256!=profile.checkpoint.sha256 || old.manifest!=profile.checkpoint.manifest
            || old.roots!=profile.checkpoint.roots{return Err("Wan resident checkpoint changed".into());}}
        Ok(profile.checkpoint.load()?)
    }
    fn inventory(&self,profile:&Profile,uploaded:Option<&BTreeMap<String,(usize,usize)>>,dispatch:&[String])->Result<Value>{
        let declaration=self.admit_source(profile)?;let mut parts=Vec::new();
        let policy=native_core_infra::weight_loader::Config::from_env()?;
        for &name in self.components(){
            let requests=preload_contract::requests(&declaration.manifest,name)?;
            let count=requests.len();let bytes=requests.iter().map(|r|declaration.manifest.tensors[&r.name].bytes).sum::<usize>();
            let host=self.host.get(name).ok_or("Wan host component missing")?;
            if host.bytes()!=bytes || host.names().count()!=count{return Err("Wan actual host inventory differs".into());}
            let mut row=json!({"name":name,"priority":preload_contract::priority(name),"manifest_sha256":profile.checkpoint.sha256,
                "tensor_count":count,"payload_bytes":bytes,"verified_tensor_count":count,"verified_payload_bytes":bytes,
                "host_ready":true,"device_uuid":self.uuid()?,"reader_policy":policy});
            if let Some(uploaded)=uploaded{
                let present=self.gpu.get(name).map(|loaded|(loaded.tensors.len(),loaded.tensors.values().map(|v|v.bytes).sum::<usize>())).unwrap_or((0,0));
                let copied=uploaded.get(name).copied().unwrap_or((0,0));
                if present!=(0,0) && present!=(count,bytes) || copied.0>present.0 || copied.1>present.1{
                    return Err("Wan actual GPU inventory differs".into());
                }
                row["resident_tensor_count"]=json!(present.0);row["resident_payload_bytes"]=json!(present.1);
                row["uploaded_tensor_count"]=json!(copied.0);row["uploaded_payload_bytes"]=json!(copied.1);
                row["reused_tensor_count"]=json!(present.0-copied.0);row["reused_payload_bytes"]=json!(present.1-copied.1);
                row["copy_complete"]=json!(present==(count,bytes));
            }
            parts.push(row);
        }
        Ok(json!({"schema":"native-preload-inventory-v1","components":parts,"dispatch_order":dispatch}))
    }
    fn warm_profile(&mut self,profile:&Profile)->Result<()> {
        let declaration=self.admit_source(profile)?;
        if self.host.is_empty(){
            let mut options=Options::for_declaration(&declaration)?;options.retained_budget=64usize<<30;options.verify=Verify::Full;
            let stages=self.components().iter().map(|n|preload_contract::requests(&declaration.manifest,n)).collect::<std::result::Result<Vec<_>,_>>()?;
            let hosts=model_load::HostPreparation::start(declaration.manifest,declaration.roots,stages,options)?.finish()?;
            if hosts.len()!=self.components().len(){return Err("Wan host preparation omitted a component".into());}
            self.host=self.components().iter().map(|n|n.to_string()).zip(hosts).collect();self.source=Some(profile.checkpoint.clone());
        }Ok(())
    }
    fn prefetch_profile(&mut self,path:&Path,budget:usize,selection:Option<&[String]>)->Result<Value>{
        let profile=Profile::read(path)?;self.warm_profile(&profile)?;let declaration=self.admit_source(&profile)?;
        let requested=selection.map(|n|n.to_vec()).unwrap_or_else(||self.components().iter().map(|n|n.to_string()).collect());
        if (requested.is_empty() && self.device.index==0) || requested.iter().any(|n|!self.components().contains(&n.as_str()))
            || requested.iter().collect::<std::collections::BTreeSet<_>>().len()!=requested.len(){return Err("Wan selected preload components differ".into());}
        let required=requested.iter().filter(|n|!self.gpu.contains_key(*n)).map(|n|{
            let requests=preload_contract::requests(&declaration.manifest,n)?;
            let sizes=requests.iter().map(|r|declaration.manifest.tensors[&r.name].bytes).collect::<Vec<_>>();
            Ok(native_core_infra::weight_loader::tensor_allocation_layout(&sizes)?.1)
        }).collect::<std::result::Result<Vec<usize>,String>>()?.into_iter().sum::<usize>();
        if required>budget{return Err("Wan preload exceeds admitted device bytes".into());}
        let mut uploaded=BTreeMap::new();let mut dispatch=Vec::new();let mut secondary=None;
        let upload_started=native_core_infra::execution_start::monotonic_ns();
        for &name in self.components(){
            // Encoder uploads precede both DiT copies. Submit the peer before
            // copying the primary DiT so the two physical transfers overlap.
            if name=="dit"{secondary=self.peer.as_ref().map(|peer|peer.begin_prefetch(path,budget,selection)).transpose()?;}
            if !requested.iter().any(|n|n==name) || self.gpu.contains_key(name){continue;}
            let requests=preload_contract::requests(&declaration.manifest,name)?;
            let loaded=unsafe{self.host[name].restore(&declaration.manifest,&requests,self.device.stream)}?;
            let bytes=loaded.tensors.values().map(|v|v.bytes).sum::<usize>();
            uploaded.insert(name.to_string(),(loaded.tensors.len(),bytes));dispatch.push(name.to_string());self.gpu.insert(name.into(),Rc::new(loaded));
        }
        let complete=self.components().iter().all(|name|self.gpu.contains_key(*name));
        if complete{self.profile_sha=Some(sha256_hex(&std::fs::read(path)?));}
        self.device.sync()?;self.generation=self.generation.checked_add(1).ok_or("Wan generation overflow")?;
        let value=json!({"generation":self.generation,"device_uuid":self.uuid()?,"world_size":1,
            "upload_started_monotonic_ns":upload_started,"upload_completed_monotonic_ns":native_core_infra::execution_start::monotonic_ns(),
            "device_bytes_uploaded":uploaded.values().map(|(_,b)|b).sum::<usize>(),"copy_complete":true,"payload_verified":true,
            "gpu_prepare_complete":complete,"gpu_prepare_required":false,"prefetch_scope":"immutable-runtime-weights",
            "host_pinned_bytes":self.host.values().map(|h|h.pinned_bytes()).sum::<usize>(),
            "reader_policy":native_core_infra::weight_loader::Config::from_env()?,
            "preload_inventory":self.inventory(&profile,Some(&uploaded),&dispatch)?});
        let peer=secondary.map(|pending|pending.finish()).transpose()?;self.combine_ranks(value,peer)
    }
    pub(crate) fn retire_predictors(&mut self)->Result<Value>{
        let secondary=self.peer.as_ref().map(|peer|peer.begin_retire()).transpose()?;
        self.device.sync()?;let before=self.device.free_bytes()?;
        {let _retirement=crate::cuda::RetirementScope::enter();
            self.dit.take();self.clip.take();self.t5.take();}
        let mut retired=0;
        for name in ["dit","clip","t5"]{if let Some(loaded)=self.gpu.remove(name){retired+=loaded.stats.device_bytes;drop(loaded);}}
        self.profile_sha=None;self.device.sync()?;
        let after=self.device.free_bytes()?;
        let mut value=json!({"retired":true,"ranks":1,"decoder_weights_preserved":self.device.index==1||(self.vae.is_some()&&self.gpu.contains_key("vae")),
            "host_weights_preserved":self.host.len()==self.components().len(),"retired_allocation_bytes":retired,
            "retired_private_scratch_bytes":after.saturating_sub(before).saturating_sub(retired),
            "devices":[{"device_uuid":self.uuid()?,"free_before_bytes":before,"free_after_bytes":after}]});
        if let Some(secondary)=secondary{
            let peer=secondary.finish()?;
            if peer["retired"]!=true||peer["host_weights_preserved"]!=true{return Err("Wan CFG retirement incomplete".into());}
            value["ranks"]=json!(2);value["all_ranks_completed"]=json!(true);
            value["retired_allocation_bytes"]=json!(retired+peer["retired_allocation_bytes"].as_u64().ok_or("Wan peer retired bytes missing")? as usize);
            value["devices"].as_array_mut().ok_or("Wan retirement devices")?.extend(peer["devices"].as_array().ok_or("Wan peer retirement devices")?.iter().cloned());
        }Ok(value)
    }
}
impl ResidentSession for Session{
    fn metadata(&self)->Result<Value>{
        let bytes=self.gpu.values().map(|v|v.stats.device_bytes).sum::<usize>();
        let row=json!({"device_uuid":self.uuid()?,"device_bytes":bytes,"device_free_bytes":self.device.free_bytes()?,"startup_loader":"core_infra::model_load"});
        let mut value=json!({"device_uuid":self.uuid()?,"world_size":self.world(),"device_bytes":bytes,"host_pinned_bytes":self.host.values().map(|h|h.pinned_bytes()).sum::<usize>(),
            "startup_loader":"core_infra::model_load","devices":[row],"resident_components":self.gpu.keys().collect::<Vec<_>>()});
        if let Some(peer)=&self.peer{
            let peer=peer.status()?;value["devices"].as_array_mut().ok_or("Wan devices")?.push(peer["devices"][0].clone());
            value["host_pinned_bytes"]=json!(value["host_pinned_bytes"].as_u64().unwrap_or(0)+peer["host_pinned_bytes"].as_u64().unwrap_or(0));
        }Ok(value)
    }
    fn warm(&mut self,path:&Path)->Result<Value>{
        let secondary=self.peer.as_ref().map(|peer|peer.begin_warm(path)).transpose()?;
        let profile=Profile::read(path)?;self.warm_profile(&profile)?;
        let peer=secondary.map(|pending|pending.finish()).transpose()?;
        let mut value=self.metadata()?;
        // Rank-zero quantities stay local; combine_ranks sums the physical receipts.
        value["host_pinned_bytes"]=json!(self.host.values().map(|h|h.pinned_bytes()).sum::<usize>());
        value["host_weights_ready"]=json!(true);value["reader_policy"]=json!(native_core_infra::weight_loader::Config::from_env()?);
        value["preload_inventory"]=self.inventory(&profile,None,&[])?;self.combine_ranks(value,peer)
    }
    fn prefetch(&mut self,path:&Path,budget:usize)->Result<Value>{self.prefetch_components(path,budget,None)}
    fn prefetch_components(&mut self,path:&Path,budget:usize,names:Option<&[String]>)->Result<Value>{
        self.prefetch_profile(path,budget,names)
    }
    fn evict(&mut self)->Result<Value>{
        self.device.sync()?;self.dit.take();self.clip.take();self.t5.take();self.vae.take();self.gpu.clear();
        self.profile_sha=None;self.device.release_idle();if let Some(peer)=&self.peer{peer.evict()?;}self.metadata()
    }
    fn audit(&mut self)->Result<Value>{
        let source=self.source.as_ref().ok_or("Wan checkpoint missing")?.load()?;
        let mut count=0;self.device.sync()?;
        let upload=native_core_infra::weight_loader::device::Upload::new(self.device.stream,native_core_infra::weight_loader::Config::from_env()?)?;
        for loaded in self.gpu.values(){for(name,view)in &loaded.tensors{
            unsafe{upload.verify(view.ptr as *mut _,view.bytes,&source.manifest.tensors[name].sha256)}?;count+=1;
        }}
        if let Some(peer)=&self.peer{let result=peer.audit()?;
            if result["full_device_payload_sha256_verified"]!=true{return Err("Wan peer audit failed".into());}
            count+=result["tensors"].as_u64().ok_or("Wan peer audit count")? as usize;
        }
        Ok(json!({"full_device_payload_sha256_verified":true,"tensors":count}))
    }
    fn infer(&mut self,profile_path:&Path,input:&Path,output:&Path,context:&Context)->Result<Value>{
        context.checkpoint()?;let profile=Profile::read(profile_path)?;
        if self.profile_sha.as_deref()!=Some(&sha256_hex(&std::fs::read(profile_path)?)) || self.gpu.len()!=4{return Err("Wan inference has no complete prepared weights".into());}
        let input:Value=serde_json::from_slice(&std::fs::read(input)?)?;
        if input["schema"]!="wan21-request-v1"{return Err("Wan raw request schema differs".into());}
        let png=std::fs::read(input["image"]["path"].as_str().ok_or("Wan image path missing")?)?;
        if input["image"]["sha256"].as_str()!=Some(&sha256_hex(&png)){return Err("Wan raw image SHA differs".into());}
        let prepared=frontend::prepare_request(Request::from_json(&input["request"])? ,&png,&profile.tokenizer.path)?;
        if profile.geometry!=[prepared.height(),prepared.width(),prepared.request.frame_num]{return Err("Wan request profile geometry differs".into());}
        let device:&'static Device=unsafe{&*(&*self.device as *const Device)};
        let part=|name:&str|DeviceWeights{loaded:self.gpu[name].clone(),prefix:format!("{name}.")};
        if self.t5.is_none(){self.t5=Some(T5::load(&part("t5"),device)?);}
        if self.clip.is_none(){self.clip=Some(ClipVisual::load(&part("clip"),device)?);}
        if self.vae.is_none(){self.vae=Some(Vae::load(&part("vae"),device)?);}
        if self.dit.is_none(){self.dit=Some(Dit::load_from(&part("dit"),pipeline::grid(&prepared),device)?);}
        let capture=native_core_infra::execution_start::Capture::begin();let mut checks=Vec::new();
        let peer_before=self.peer.as_ref().map(|peer|peer.status()).transpose()?
            .and_then(|value|value["unconditional_forwards"].as_u64()).unwrap_or(0);
        let observer=self.observer.as_ref();
        let mut observe=|name:&str,address:u64|->Result<()>{
            context.checkpoint()?;
            if let Some(entry)=observer.and_then(|g|g.tensors.get(name)){
                device.sync()?;let data=Buffer::view(address,entry.bytes)?.download(device)?;let actual=sha256_hex(&data);
                if actual!=entry.sha256{return Err(format!("Wan gold mismatch: {name}").into());}
                checks.push(json!({"name":name,"sha256":actual,"expected_sha256":entry.sha256,"equal":true}));
            }Ok(())
        };
        let began=Instant::now();context.phase("conditioning");
        // A verified request computes (and checks) its conditioning; others may reuse the last one.
        let key=self.observer.is_none().then(||pipeline::conditioning_key(&prepared,self.profile_sha.as_deref().unwrap_or("")));
        let cond=match (key.as_ref(),self.conditioning.take()){
            (Some(key),Some((cached,cond))) if *key==cached=>cond,
            _=>pipeline::condition(self.t5.as_ref().unwrap(),self.clip.as_ref().unwrap(),self.vae.as_ref().unwrap(),device,&prepared,&mut observe)?,
        };
        context.phase("sampling");let latent=pipeline::denoise_with_peer(self.dit.as_mut().unwrap(),device,&prepared,&cond,&mut observe,self.peer.as_ref())?;
        if let Some(key)=key{self.conditioning=Some((key,cond));}else{drop(cond);}
        device.sync()?;let latent_sha=sha256_hex(&latent.download(device)?);
        // End the observer borrow before the actor mutates its component owners.
        drop(observe);
        let handoff=context.before_gpu_phase_with_retirement("decode",&profile.geometry,||self.retire_predictors())?;
        let vae_submit=native_core_infra::execution_start::monotonic_ns();let mut vae_start=None;
        let mut observe=|name:&str,address:u64|->Result<()>{
            context.checkpoint()?;
            if let Some(entry)=self.observer.as_ref().and_then(|g|g.tensors.get(name)){
                device.sync()?;let actual=sha256_hex(&Buffer::view(address,entry.bytes)?.download(device)?);
                if actual!=entry.sha256{return Err(format!("Wan gold mismatch: {name}").into());}
                checks.push(json!({"name":name,"sha256":actual,"expected_sha256":entry.sha256,"equal":true}));
            }Ok(())
        };
        let(video,frames)=pipeline::decode_with_start(self.vae.as_ref().unwrap(),&prepared,&latent,&mut observe,|stream|{
            vae_start=Some(unsafe{native_core_infra::execution_start::StreamTimestamp::record(stream)}?);Ok(())
        })?;
        drop(observe);let vae_end=unsafe{native_core_infra::execution_start::StreamTimestamp::record(device.stream)}?;
        device.sync()?;let vae_start=vae_start.ok_or("VAE start marker missing")?.read()?;let vae_end=vae_end.read()?;
        let primary_first=capture.report()?.ok_or("Wan native DiT marker disabled")?;
        let mut first_dit_ranks=vec![primary_first];
        let steps=native_core_infra::switch_benchmark::steps(prepared.request.sample_steps,false)?;
        if let Some(peer)=&self.peer{
            let status=peer.status()?;
            if status["unconditional_forwards"].as_u64().and_then(|n|n.checked_sub(peer_before))!=Some(steps as u64){
                return Err("Wan GPU 1 did not execute every CFG pass".into());
            }
            if status["first_dit_start"]["monotonic_ns"].as_u64().is_none(){return Err("Wan GPU 1 first DiT marker missing".into());}
            first_dit_ranks.push(status["first_dit_start"].clone());
        }
        let first_dit=first_dit_ranks.iter().min_by_key(|v|v["monotonic_ns"].as_u64().unwrap_or(u64::MAX)).unwrap().clone();
        let bytes=video.download(device)?;let raw_sha=sha256_hex(&bytes);
        if let Some(gold)=&self.observer{if gold.tensors.get("video/out").is_some_and(|e|e.sha256!=raw_sha){return Err("Wan decoded video gold mismatch".into());}}
        if output.exists(){return Err("Wan output directory must be fresh".into());}std::fs::create_dir_all(output)?;
        let generated=pipeline::Generated{video,frames,timings:pipeline::Timings::default()};
        let mut gpu_complete=None;
        pipeline::write_mp4_after_download(&generated,&prepared,device,&output.join("output.mp4"),||{
            context.checkpoint()?;
            gpu_complete=Some(native_core_infra::execution_start::monotonic_ns());
            context.phase("decode_tail");Ok(())
        })?;
        Ok(json!({"first_dit_start":first_dit,"latent_sha256":latent_sha,"video_raw_sha256":raw_sha,
            "first_dit_ranks":first_dit_ranks,
            "parallel":{"world_size":self.world(),"all_ranks_completed":true,"forwards_per_rank":if self.peer.is_some(){vec![steps,steps]}else{vec![steps*2]},
                "devices":self.metadata()?["devices"]},
            "height":prepared.height(),"width":prepared.width(),"frames":frames,"checks":checks,
            "sampling_steps":native_core_infra::switch_benchmark::steps(prepared.request.sample_steps,false)?,
            "vae_submit_monotonic_ns":vae_submit,"vae_started_monotonic_ns":vae_start,"vae_completed_monotonic_ns":vae_end,
            "gpu_work_completed_monotonic_ns":gpu_complete.ok_or("Wan media tail boundary missing")?,
            "decode_handoff":handoff,"generate_seconds":began.elapsed().as_secs_f64(),"reference_activations_injected":false}))
    }
    fn verify(&mut self,config:Value,context:&Context)->Result<Value>{
        if std::env::var_os("NATIVE_SWITCH_BENCH_STEPS").is_some(){return Err("Wan full numerical gold refuses a benchmark cap".into());}
        let profile:Pin=serde_json::from_value(config["profile"].clone())?;profile.verify()?;
        let input:Pin=serde_json::from_value(config["request"].clone())?;input.verify()?;
        let gold:Pin=serde_json::from_value(config["gold"].clone())?;gold.verify()?;
        self.observer=Some(Gold::open(gold.path.parent().ok_or("gold root missing")?)?);
        let result=self.infer(&profile.path,&input.path,Path::new(config["output"].as_str().ok_or("gold output missing")?),context);
        self.observer=None;result
    }
}
impl Drop for Session{fn drop(&mut self){let _=self.device.sync();self.dit.take();self.clip.take();self.t5.take();self.vae.take();self.gpu.clear();self.host.clear();}}
