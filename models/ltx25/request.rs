//! Compose native text, request noise, two AV sampling stages and the stage
//! bridge. Recorded activations are comparison targets, never successors.
use crate::{cuda::{Device,upload_into},gold::{Gold,sha},graph,linear::Linear,rng::CudaGenerator,
    sampling::{self,State},tensor::{self,Tensor},text,weights::Checkpoint,Result};
use serde::Deserialize;
use serde_json::{json,Value};
use std::{cell::{Cell,RefCell},collections::BTreeMap,path::{Path,PathBuf},rc::Rc,sync::Arc};

#[derive(Deserialize)]
pub struct Config {
    pub gold:PathBuf,pub forward:PathBuf,pub text:PathBuf,pub upsampler:PathBuf,
    pub weights:PathBuf,pub output:PathBuf,
    #[serde(default)] pub encoder:Option<PathBuf>,
    #[serde(default)] pub raw_image:Option<PathBuf>,
    #[serde(default)] pub front_only:bool,
    #[serde(default)] pub inference:bool,
    #[serde(default)] pub profile_manifests:BTreeMap<String,String>,
    #[serde(default)] pub raw_request:Option<PathBuf>,
    #[serde(default)] pub audio:Option<PathBuf>,
    #[serde(default)] pub video:Option<PathBuf>,
}
fn cast(x:&Tensor,dtype:&str,device:&Device)->Result<Tensor>{
    let out=Tensor::contiguous_shape(x.shape.clone(),dtype.into(),device)?;
    tensor::copy(x,&out,device.stream)?;Ok(out)
}
fn filled(shape:Vec<usize>,dtype:&str,value:f32,device:&Device)->Result<Tensor>{
    let out=Tensor::contiguous_shape(shape,dtype.into(),device)?;
    tensor::fill(&out,value,0.,device.stream)?;Ok(out)
}
fn compare(gold:&Gold,name:&str,value:&Tensor,device:&Device,checks:&mut Vec<Value>)->Result<()>{
    if !gold.verify{return Ok(());}
    let entry=gold.entry(name)?;let actual=sha(&value.download(device)?);
    let passed=actual==entry.sha256 && value.shape==entry.shape && value.dtype==entry.dtype;
    checks.push(json!({"tensor":name,"passed":passed,"sha256":actual,"expected_sha256":entry.sha256}));
    if !passed{return Err(format!("native request differs at {name}: {actual} != {}",entry.sha256).into());}Ok(())
}
/// Reuses verified immutable checkpoint identities and native handles. A
/// resident worker owns this value on one thread; request activations remain
/// local to run(). Field order keeps CUDA alive until its dependents are gone.
pub struct Session {
    declared:Option<RefCell<crate::declared::LtxStartup>>,
    framework:Option<Arc<native_core_infra::checkpoint::Declaration>>,
    checkpoints:RefCell<BTreeMap<PathBuf,(String,Rc<Checkpoint>)>>,
    host_budget:native_core_infra::pinned_weights::HostBudget,
    assets:Rc<native_core_infra::immutable_assets::ImmutableAssets>,
    linear:Rc<Linear>,device:Rc<Device>,resident:bool,retain_host_cache:bool,generation:Cell<u64>,
    linear_shapes_prepared:usize,processor_attention_prepared:Rc<RefCell<BTreeMap<PathBuf,usize>>>,
    cpu_preparation:RefCell<Option<native_core_infra::startup_policy::CpuPreparation<crate::frontend::GemmaTokenizer>>>,
    pending:RefCell<Option<native_core_infra::model_load::Components>>,
    checkpoint_identity:Option<String>,
    peer:Option<crate::parallel::ParallelBlocks>,
    prepared_profile:RefCell<Option<(PathBuf,String)>>,
    prepared_receipt:RefCell<Option<Value>>,
    phase_callback:Option<Box<dyn Fn(&str)>>,
}
impl Session {
    pub fn open()->Result<Self>{
        let device=Rc::new(Device::open()?);let linear=Rc::new(Linear::new(&device)?);
        Ok(Self {declared:None,cpu_preparation:RefCell::new(None),pending:RefCell::new(None),peer:None,checkpoint_identity:None,prepared_profile:RefCell::new(None),prepared_receipt:RefCell::new(None),framework:None,checkpoints:RefCell::new(BTreeMap::new()),
            host_budget:native_core_infra::pinned_weights::HostBudget::new(80usize<<30),
            assets:Rc::new(native_core_infra::immutable_assets::ImmutableAssets::new(2usize<<30,4096)),
            linear,device,resident:false,retain_host_cache:true,generation:Cell::new(0),linear_shapes_prepared:0,
            processor_attention_prepared:Rc::new(RefCell::new(BTreeMap::new())),phase_callback:None})
    }
    pub fn open_resident()->Result<Self>{
        Self::open_resident_world(1)
    }
    pub fn open_resident_world(world:usize)->Result<Self>{
        let framework=crate::admission::checkpoint(None)?;
        if world==1{Self::open_declared(framework,true)}else{Self::open_parallel(framework,true,world)}
    }
    /// With two GPUs, GPU 0 loads exactly as one GPU does. GPU 1 opens its
    /// context once GPU 0's is ready (opening both at once delays GPU 0's
    /// uploads), then copies blocks 24-47 from GPU 0's DiT over the peer link.
    pub(crate) fn open_resident_with_checkpoint_world(framework:impl Into<Option<native_core_infra::checkpoint::Declaration>>,
        world:usize)->Result<Self>{
        if !matches!(world,1|2){return Err("LTX supports one or two physical GPUs".into());}
        let mut session=Self::open_declared(framework.into().ok_or("LTX checkpoint declaration required")?,true)?;
        if world==2{
            let declaration=session.framework.as_deref().map(|value|native_core_infra::checkpoint::Declaration{
                manifest:value.manifest.clone(),roots:value.roots.clone()});
            session.peer=Some(crate::parallel::ParallelBlocks::start(declaration,false)?);
        }
        Ok(session)
    }
    pub(crate) fn open_backend(framework:impl Into<Option<native_core_infra::checkpoint::Declaration>>)->Result<Self>{
        Self::open_backend_world(framework,1)
    }
    pub(crate) fn open_backend_world(framework:impl Into<Option<native_core_infra::checkpoint::Declaration>>,world:usize)->Result<Self>{
        let framework=framework.into().ok_or("LTX checkpoint declaration required")?;
        if world==1{Self::open_declared(framework,false)}else{Self::open_parallel(framework,false,world)}
    }
    fn open_declared(framework:native_core_infra::checkpoint::Declaration,activate:bool)->Result<Self>{
        let framework=Arc::new(framework);
        let mut startup=crate::declared::LtxStartup::new(framework.clone(),activate)?;
        let resources=startup.start()?;let attention=startup.attention.clone();
        Ok(Self {declared:Some(RefCell::new(startup)),cpu_preparation:RefCell::new(None),pending:RefCell::new(None),peer:None,checkpoint_identity:None,prepared_profile:RefCell::new(None),prepared_receipt:RefCell::new(None),framework:Some(framework),checkpoints:RefCell::new(BTreeMap::new()),
            host_budget:native_core_infra::pinned_weights::HostBudget::new(80usize<<30),
            assets:resources.assets,linear:resources.linear,device:resources.device,resident:true,retain_host_cache:true,generation:Cell::new(0),
            linear_shapes_prepared:resources.linear_shapes,processor_attention_prepared:attention,phase_callback:None})
    }
    fn open_parallel(framework:native_core_infra::checkpoint::Declaration,activate:bool,world_size:usize)->Result<Self>{
        if !matches!(world_size,1|2){return Err("LTX supports one or two physical GPUs".into());}
        let framework=Some(Arc::new(framework));
        let cpu_preparation=framework.as_ref().map(|value|crate::startup::prepare_cpu(value)).transpose()?;
        let pending=if activate{framework.as_ref().map(|declaration|crate::startup::start_world(declaration,world_size)).transpose()?}else{None};
        let peer_framework=framework.as_ref().map(|value|native_core_infra::checkpoint::Declaration{
            manifest:value.manifest.clone(),roots:value.roots.clone()});
        let mut session=Self::with_device(Device::open_private(0)?,true,framework,pending,cpu_preparation,activate)?;
        if world_size==2{session.peer=Some(crate::parallel::ParallelBlocks::open_with_checkpoint(peer_framework)?);}Ok(session)
    }
    fn with_device(mut device:Device,resident:bool,framework:Option<Arc<native_core_infra::checkpoint::Declaration>>,
        mut pending:Option<native_core_infra::model_load::Components>,
        cpu_preparation:Option<native_core_infra::startup_policy::CpuPreparation<crate::frontend::GemmaTokenizer>>,activate:bool)->Result<Self>{
        let assets=Rc::new(native_core_infra::immutable_assets::ImmutableAssets::new(2usize<<30,4096));
        if resident{device.host_assets=Some(assets.clone());}
        let device=Rc::new(device);let linear=Rc::new(Linear::new(&device)?);
        let mut linear_shapes_prepared=0;
        if let Some(declaration)=&framework {
            let shapes=crate::startup::linear_shapes(declaration)?;
            native_core_infra::startup_trace::measure("startup_backend_prepare","linear metadata",0,||linear.prepare(&shapes))?;
            linear_shapes_prepared=shapes.len();
        }
        if activate{if let Some(pending)=&mut pending{unsafe{pending.device_ready()}?;}}
        Ok(Self{declared:None,retain_host_cache:true,linear_shapes_prepared,processor_attention_prepared:Rc::new(RefCell::new(BTreeMap::new())),cpu_preparation:RefCell::new(cpu_preparation),pending:RefCell::new(pending),framework,checkpoint_identity:None,checkpoints:RefCell::new(BTreeMap::new()),
            host_budget:native_core_infra::pinned_weights::HostBudget::new(80usize<<30),linear,device,resident,
            assets,peer:None,
            generation:Cell::new(0),prepared_profile:RefCell::new(None),prepared_receipt:RefCell::new(None),phase_callback:None})
    }
    pub fn set_phase_callback(&mut self,callback:Box<dyn Fn(&str)>){self.phase_callback=Some(callback);}
    pub(crate) fn set_checkpoint_identity(&mut self,digest:String)->Result<()>{
        if let Some(peer)=&self.peer{peer.set_checkpoint_identity(digest.clone())?;}
        self.checkpoint_identity=Some(digest);Ok(())
    }
    fn phase(&self,name:&str){if let Some(callback)=&self.phase_callback{callback(name);}}
    fn use_component(&self,name:&str)->Result<Option<native_core_infra::declared_startup::ConsumerGuard>> {
        self.declared.as_ref().map(|runtime|runtime.borrow_mut().enter(name)).transpose().map_err(Into::into)
    }
    fn preparation_report(&self)->Value{
        json!({"required":["text linear metadata","processor attention layouts","Gemma tokenizer"],
            "text_linear_shapes":self.linear_shapes_prepared,
            "processor_attention_plans":*self.processor_attention_prepared.borrow(),
            "tokenizer_complete":self.framework.is_none()
                || self.checkpoints.borrow().values().any(|(_,checkpoint)|checkpoint.tokenizer_loaded()),
            "remaining_first_use":["Triton module loading","native convolution plans and workspaces","request activation and sampler allocations"],
            "full_execution_prepared":false})
    }
    fn finish_cpu_preparation(&self,text:&Checkpoint)->Result<()>{
        if let Some(preparation)=self.cpu_preparation.borrow_mut().take(){text.install_tokenizer(preparation.finish()?)?;}Ok(())
    }
    fn prepare_text_backend(&self,text:&Gold)->Result<()>{
        let _prepare=native_core_infra::startup_trace::Span::new("startup_backend_prepare","processor attention metadata",0);
        let processor:Value=serde_json::from_slice(&std::fs::read(text.root.join("processor.graph.json"))?)?;
        let count=graph::prepare_attention(&processor)?;
        self.processor_attention_prepared.borrow_mut().insert(text.root.clone(),count);Ok(())
    }
    fn checkpoint(&self,gold:&Gold,root:&Path,kind:&str,file:&str)->Result<Rc<Checkpoint>>{
        let expected=gold.identity["weights"][kind]["sha256"].as_str().ok_or("weight identity")?;
        let path=root.join(file);
        if let Some((sha,checkpoint))=self.checkpoints.borrow().get(&path){
            if sha!=expected{return Err("resident checkpoint identity changed".into());}
            return Ok(checkpoint.clone());
        }
        let _admit=native_core_infra::startup_trace::Span::new("startup_request",&format!("checkpoint {kind}"),0);
        let mut checkpoint=Checkpoint::open(&path)?;
        let prefix=match kind {"transformer"=>Some("dit"),"text_encoder"=>Some("text"),"video_vae"=>Some("vae"),
            "spatial_upsampler"=>Some("upscale"),_=>None};
        if let (Some(declaration),Some(prefix))=(&self.framework,prefix){
            checkpoint.admit(prefix,expected,std::sync::Arc::new(declaration.manifest.clone()),&declaration.roots)?;
        }else if self.resident{checkpoint.pin(expected,&self.host_budget)?;}else{checkpoint.verify(expected)?;}
        let checkpoint=Rc::new(checkpoint);
        self.checkpoints.borrow_mut().insert(path,(expected.into(),checkpoint.clone()));Ok(checkpoint)
    }
    /// Load only original model tensors. The caller reserves decoder workspace
    /// before granting max_device_bytes; no sampling or tensor conversion runs.
    pub fn prefetch(&self,profile:&Path,max_device_bytes:usize)->Result<Value>{
        self.prepare_weights(profile,max_device_bytes,true)
    }
    fn prepare_weights(&self,profile:&Path,max_device_bytes:usize,copy_ticket:bool)->Result<Value>{
        self.prepare_components(profile,max_device_bytes,copy_ticket,None)
    }
    pub(crate) fn prefetch_components(&self,profile:&Path,budget:usize,components:&[String])->Result<Value>{
        let names=["text_encoder","video_vae","dit","spatial_upsampler"];
        if components.is_empty() || components.iter().any(|n|!names.contains(&n.as_str())) ||
            components.iter().collect::<std::collections::HashSet<_>>().len()!=components.len(){
            return Err("LTX partial preload requires unique declared component IDs".into());
        }
        self.prepare_components(profile,budget,true,Some(components))
    }
    fn prepare_components(&self,profile:&Path,max_device_bytes:usize,copy_ticket:bool,selected:Option<&[String]>)->Result<Value>{
        let begun=std::time::Instant::now();
        if !self.resident{return Err("prefetch requires a resident worker".into());}
        let raw=std::fs::read(profile)?;let profile_sha=sha(&raw);
        if self.prepared_profile.borrow().as_ref().is_some_and(|(path,digest)|path==profile && *digest==profile_sha){
            // No new weights, layout or request workspace is needed. The
            // immutable lease is invalidated by every cache retirement path.
            self.device.sync()?;
            let generation=if copy_ticket{self.generation.get().checked_add(1).ok_or("weight generation overflow")?}
                else{self.generation.get()};
            self.generation.set(generation);
            let mut receipt=self.prepared_receipt.borrow().as_ref().ok_or("prepared weight receipt missing")?.clone();
            receipt["generation"]=json!(generation);receipt["device_bytes_uploaded"]=json!(0);
            receipt["reused"]=json!(true);receipt["residency_lease_reused"]=json!(true);
            receipt["allocation_and_copy_seconds"]=json!(0.);
            receipt["parallel_preparation"]=Value::Null;
            if let Some(inventory)=receipt.get_mut("preload_inventory"){
                inventory["dispatch_order"]=json!([]);
                if let Some(rows)=inventory["components"].as_array_mut(){for row in rows{
                    row["uploaded_tensor_count"]=json!(0);row["uploaded_payload_bytes"]=json!(0);
                    row["reused_tensor_count"]=row["tensor_count"].clone();
                    row["reused_payload_bytes"]=row["payload_bytes"].clone();
                }}
            }
            if let Some(ranks)=receipt.get_mut("rank_receipts").and_then(Value::as_array_mut){for rank in ranks{
                rank["device_bytes_uploaded"]=json!(0);
                rank["preload_inventory"]["dispatch_order"]=json!([]);
                if let Some(rows)=rank["preload_inventory"]["components"].as_array_mut(){for row in rows{
                    row["uploaded_tensor_count"]=json!(0);row["uploaded_payload_bytes"]=json!(0);
                    row["reused_tensor_count"]=row["resident_tensor_count"].clone();
                    row["reused_payload_bytes"]=row["resident_payload_bytes"].clone();
                }}
            }}
            receipt["host_pinned_bytes"]=json!(self.pinned_host_bytes()?);
            receipt["devices"]=json!(self.devices()?);
            receipt["complete_prepare_seconds"]=json!(begun.elapsed().as_secs_f64());
            return Ok(receipt);
        }
        let config:Config=serde_json::from_slice(&raw)?;
        if !config.inference{return Err("prefetch requires an exported inference profile".into());}
        if self.checkpoint_identity.is_some() && self.pending.borrow().is_some(){
            return Err("registered LTX preload requires its verified host-warm path before device preparation".into());
        }
        if self.framework.is_some() && self.pending.borrow().is_none() && self.retained_bytes()==0{
            self.warm_host(profile)?;
        }
        let gold=Gold::profile(&config.gold,config.profile_manifests.get("gold").ok_or("profile identity")?)?;
        let mut groups=Vec::new();let mut tensor_count=0usize;let mut device_bytes=0usize;
        for (kind,file) in [
            ("transformer","diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
            ("text_encoder","text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
            ("video_vae","vae/ltx-2.5-video-vae-bf16.safetensors"),
            ("spatial_upsampler","latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")]{
            let checkpoint=self.checkpoint(&gold,&config.weights,kind,file)?;
            let mut missing=Vec::new();
            for (key,info) in checkpoint.header.as_object().ok_or("checkpoint header")?{
                if self.peer.is_some() && kind=="transformer" && crate::parallel::second_partition(key){continue;}
                if !matches!(info["dtype"].as_str(),Some("BF16"|"F32")){continue;}
                let offsets=info["data_offsets"].as_array().ok_or("weight offsets")?;
                if offsets.len()!=2{return Err("weight offset pair".into());}
                let bytes=offsets[1].as_u64().and_then(|end|end.checked_sub(offsets[0].as_u64()?))
                    .and_then(|bytes|usize::try_from(bytes).ok()).ok_or("weight extent")?;
                let cached=self.device.cached_weight(&format!("{}\0{key}",checkpoint.path.display())).is_some();
                let name=match kind{"transformer"=>"dit","text_encoder"=>"text_encoder","video_vae"=>"video_vae",_=>"spatial_upsampler"};
                if !cached && selected.is_none_or(|names|names.iter().any(|n|n==name)){
                    device_bytes=device_bytes.checked_add(bytes).ok_or("prefetch extent overflow")?;missing.push(key.clone());
                }
                tensor_count+=1;
            }
            groups.push((checkpoint,missing));
        }
        if device_bytes>max_device_bytes{return Err("prefetch exceeds the decoder's device memory reservation".into());}
        self.phase("prefetch");
        let upload_started=std::time::Instant::now();
        let upload_started_ns=native_core_infra::execution_start::monotonic_ns();
        let mut peer=None;let mut primary_dit_upload=None;let mut dispatch=Vec::new();
        if self.declared.is_some(){
        let text=Gold::profile(&config.text,config.profile_manifests.get("text").ok_or("text profile identity")?)?;
        let runtime=self.declared.as_ref().ok_or("resident startup declaration missing")?;
        let checkpoints=Self::framework_groups(&groups.iter().map(|(c,_)|c.clone()).collect::<Vec<_>>());
        runtime.borrow_mut().request(text.root.clone(),checkpoints)?;
        for component in ["text","connectors","dit","vae","upscale"] {drop(self.use_component(component)?);}
        }else{
        if let Some(mut pending)=self.pending.borrow_mut().take(){
            let checkpoints=Self::framework_groups(&groups.iter().map(|(c,_)|c.clone()).collect::<Vec<_>>());
            unsafe{pending.device_ready()}?;
            let text=Gold::profile(&config.text,config.profile_manifests.get("text").ok_or("text profile identity")?)?;
            self.prepare_text_backend(&text)?;
            crate::startup::install(unsafe{pending.finish_first(self.device.stream)}?,&checkpoints,&self.device)?;
            let secondary=self.peer.as_ref().map(|p|p.begin_prefetch_components(profile,max_device_bytes,
                selected.is_none_or(|names|names.iter().any(|n|n=="dit")))).transpose()?;
            for (loaded,_) in pending.spawn_rest()?.join().map_err(|_|"LTX stage thread panicked")?? {
                crate::startup::install(loaded,&checkpoints,&self.device)?;
            }
            if let Some(secondary)=secondary{peer=Some(secondary.finish()?);}
        }else{
            if !groups[1].1.is_empty(){dispatch.push("text_encoder");}
            groups[1].0.prefetch_weights(&groups[1].1,&self.device)?;
            if config.encoder.is_some(){
                if !groups[2].1.is_empty(){dispatch.push("video_vae");}
                groups[2].0.prefetch_weights(&groups[2].1,&self.device)?;
            }
            let secondary=self.peer.as_ref().map(|p|p.begin_prefetch_components(profile,max_device_bytes,
                selected.is_none_or(|names|names.iter().any(|n|n=="dit")))).transpose()?;
            let dit_started=native_core_infra::execution_start::monotonic_ns();
            if !groups[0].1.is_empty(){dispatch.push("dit");}
            groups[0].0.prefetch_weights(&groups[0].1,&self.device)?;
            let dit_completed=native_core_infra::execution_start::monotonic_ns();
            primary_dit_upload=Some(json!({"device_index":0,"upload_started_monotonic_ns":dit_started,
                "upload_completed_monotonic_ns":dit_completed}));
            if let Some(secondary)=secondary{peer=Some(secondary.finish()?);}
            if config.encoder.is_none(){
                if !groups[2].1.is_empty(){dispatch.push("video_vae");}
                groups[2].0.prefetch_weights(&groups[2].1,&self.device)?;
            }
            if !groups[3].1.is_empty(){dispatch.push("spatial_upsampler");}
            groups[3].0.prefetch_weights(&groups[3].1,&self.device)?;
        }
        if peer.is_none(){if let Some(secondary)=&self.peer{peer=Some(secondary.prefetch(profile,max_device_bytes)?);}}
        }
        self.device.sync()?;let gpu_upload_seconds=upload_started.elapsed().as_secs_f64();
        // GPU readiness does not wait for a second complete retained-RAM read.
        // run() prepares/pins that cache during sampling and joins it before
        // returning the first request, so subsequent eviction/reload is ready.
        let generation=if copy_ticket || device_bytes>0{
            self.generation.get().checked_add(1).ok_or("weight generation overflow")?
        }else{self.generation.get()};
        self.generation.set(generation);*self.prepared_profile.borrow_mut()=Some((profile.into(),profile_sha));self.phase("idle");
        let uploaded=device_bytes+peer.as_ref().and_then(|v|v["device_bytes_uploaded"].as_u64()).unwrap_or(0) as usize;
        let mut receipt=json!({"owner_pid":std::process::id(),"device_uuid":self.device.uuid()?,"generation":generation,
            "upload_started_monotonic_ns":upload_started_ns,"upload_completed_monotonic_ns":native_core_infra::execution_start::monotonic_ns(),
            "complete_prepare_seconds":begun.elapsed().as_secs_f64(),
            "allocation_and_copy_seconds":gpu_upload_seconds,
            "component_priority":if config.encoder.is_some(){vec!["text_encoder","image_encoder","dit_all_ranks","stage_bridge"]}
                else{vec!["text_encoder","dit_all_ranks","video_decoder","stage_bridge"]},
            "reader_policy":native_core_infra::weight_loader::Config::from_env()?,
            "parallel_preparation":{"primary_dit":primary_dit_upload,"secondary_dit":peer},
            "device_bytes_uploaded":uploaded,"reused":uploaded==0,
            "weight_residency":"retain until explicit eviction",
            "devices":self.devices()?,"world_size":if self.peer.is_some(){2}else{1},
            "tensors":tensor_count,"payload_verified":true,
            "allocation":"aligned owned slabs with independent checkpoint eviction",
            "verification_policy":if self.framework.is_some(){std::env::var("NATIVE_LOAD_VERIFY").unwrap_or("light".into())}else{"full".into()},
            "host_pinned_bytes":self.pinned_host_bytes()?,"host_retained_bytes":self.retained_bytes(),
            "copy_complete":true,"gpu_prepare_complete":true,"gpu_prepare_required":self.declared.is_some(),"preparation":self.preparation_report(),
            "scope":"original text, sampling, image-encoder and upsampler tensors"});
        if self.checkpoint_identity.is_some(){
            let missing=groups.iter().map(|(_,keys)|keys.clone()).collect::<Vec<_>>();
            receipt["preload_inventory"]=self.preload_inventory(&config,&gold,Some(&missing))?;
            receipt["preload_inventory"]["dispatch_order"]=json!(dispatch);
            let complete=receipt["preload_inventory"]["components"].as_array().ok_or("component inventory")?
                .iter().all(|row|row["copy_complete"]==true) && peer.as_ref().is_none_or(|value|
                    value["preload_inventory"]["components"].as_array().is_some_and(|rows|rows.iter().all(|row|row["copy_complete"]==true)));
            receipt["gpu_prepare_complete"]=json!(complete);
            receipt["preload_components_complete"]=json!(complete);
            if let Some(mut secondary)=peer{
                let mut primary=receipt.clone();primary.as_object_mut().ok_or("LTX receipt")?.remove("parallel_preparation");
                primary["device_bytes_uploaded"]=json!(device_bytes);
                secondary["gpu_prepare_complete"]=json!(complete);
                receipt["rank_receipts"]=json!([primary,secondary]);receipt["all_ranks_completed"]=json!(true);
            }
            if !complete{self.prepared_profile.borrow_mut().take();}
        }
        if receipt["gpu_prepare_complete"]==true{*self.prepared_receipt.borrow_mut()=Some(receipt.clone());}Ok(receipt)
    }
    fn preload_inventory(&self,config:&Config,gold:&Gold,missing:Option<&Vec<Vec<String>>>)->Result<Value>{
        let digest=self.checkpoint_identity.as_ref().ok_or("LTX checkpoint identity missing")?;
        let definitions=[("dit","transformer","diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
            ("text_encoder","text_encoder","text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
            ("video_vae","video_vae","vae/ltx-2.5-video-vae-bf16.safetensors"),
            ("spatial_upsampler","spatial_upsampler","latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")];
        let order=if config.encoder.is_some(){[1usize,2,0,3]}else{[1usize,0,2,3]};
        let mut components=Vec::new();let mut dispatch=Vec::new();
        for (priority,index) in order.into_iter().enumerate(){
            let(name,kind,file)=definitions[index];let checkpoint=self.checkpoint(gold,&config.weights,kind,file)?;
            let mut count=0usize;let mut bytes=0usize;let mut resident_count=0usize;let mut resident_bytes=0usize;
            let mut uploaded_count=0usize;let mut uploaded_bytes=0usize;
            let uploaded_keys=missing.map(|groups|groups[index].iter().cloned().collect::<std::collections::HashSet<_>>());
            for(key,info)in checkpoint.header.as_object().ok_or("checkpoint header")?{
                if self.peer.is_some() && name=="dit" && crate::parallel::second_partition(key){continue;}
                if !matches!(info["dtype"].as_str(),Some("BF16"|"F32")){continue;}
                let offsets=info["data_offsets"].as_array().ok_or("weight offsets")?;
                let size=offsets[1].as_u64().and_then(|n|n.checked_sub(offsets[0].as_u64()?)).ok_or("weight extent")? as usize;
                count+=1;bytes=bytes.checked_add(size).ok_or("component extent overflow")?;
                if let Some(value)=self.device.cached_weight(&format!("{}\0{key}",checkpoint.path.display())){
                    if value.elements()?.checked_mul(crate::tensor::dtype(&value.dtype)?.1)!=Some(size){
                        return Err("LTX resident tensor extent differs from checkpoint".into());
                    }
                    resident_count+=1;resident_bytes+=size;
                }
                if uploaded_keys.as_ref().is_some_and(|keys|keys.contains(key)){uploaded_count+=1;uploaded_bytes+=size;}
            }
            if checkpoint.retained_bytes()==0{return Err("LTX declared host component is not retained".into());}
            let mut row=json!({"name":name,"priority":priority,"manifest_sha256":digest,"tensor_count":count,
                "payload_bytes":bytes,"reader_policy":native_core_infra::weight_loader::Config::from_env()?,
                "host_ready":true,"device_uuid":self.device.uuid()?,"verified_tensor_count":count,"verified_payload_bytes":bytes});
            if missing.is_some(){
                row["resident_tensor_count"]=json!(resident_count);row["resident_payload_bytes"]=json!(resident_bytes);
                row["uploaded_tensor_count"]=json!(uploaded_count);row["uploaded_payload_bytes"]=json!(uploaded_bytes);
                row["reused_tensor_count"]=json!(resident_count-uploaded_count);row["reused_payload_bytes"]=json!(resident_bytes-uploaded_bytes);
                row["copy_complete"]=json!(count==resident_count && bytes==resident_bytes);if uploaded_bytes>0{dispatch.push(name);}
            }
            components.push(row);
        }
        Ok(json!({"schema":"native-preload-inventory-v1","scope":"before_first_dit",
            "components":components,"dispatch_order":dispatch}))
    }
    fn framework_groups(checkpoints:&[Rc<Checkpoint>])->Vec<(&'static str,Rc<Checkpoint>)>{
        ["dit","text","vae","upscale"].into_iter().zip(checkpoints.iter().cloned()).collect()
    }
    fn retained_bytes(&self)->usize {self.checkpoints.borrow().values().map(|(_,c)|c.retained_bytes()).sum()}
    fn pinned_host_bytes(&self)->Result<usize>{
        let mut seen=std::collections::HashSet::new();let mut bytes=self.host_budget.used();
        for (_,checkpoint) in self.checkpoints.borrow().values(){for host in checkpoint.retained_hosts(){
            if seen.insert(std::sync::Arc::as_ptr(&host) as usize){bytes+=host.pinned_bytes();}
        }}
        if let Some(peer)=&self.peer{bytes+=peer.status()?["host_pinned_bytes"].as_u64().ok_or("secondary pinned host accounting")? as usize;}
        Ok(bytes)
    }
    pub fn evict(&self)->Result<()>{
        self.device.clear_weight_cache()?;if let Some(peer)=&self.peer{peer.evict()?;}
        self.prepared_profile.borrow_mut().take();self.prepared_receipt.borrow_mut().take();Ok(())
    }
    fn devices(&self)->Result<Vec<Value>>{
        let mut values=vec![json!({"device_index":0,"device_uuid":self.device.uuid()?,
            "cached_weight_tensors":self.device.cached_weight_tensors(),
            "owner_pid":std::process::id(),"owner_instance":native_core_infra::process_identity::current(),
            "first_block":0,"end_block":if self.peer.is_some(){24}else{48}})];
        if let Some(peer)=&self.peer{values.push(peer.status()?);}Ok(values)
    }
    fn pin_retained_sources(&self)->Result<()> {
        let mut seen=std::collections::HashSet::new();
        for (_,checkpoint) in self.checkpoints.borrow().values(){for host in checkpoint.retained_hosts(){
            if seen.insert(std::sync::Arc::as_ptr(&host) as usize){host.pin_for_use()?;}
        }}Ok(())
    }
    pub fn metadata(&self)->Result<Value>{
        Ok(json!({"owner_pid":std::process::id(),"device_uuid":self.device.uuid()?,
            "host_pinned_bytes":self.pinned_host_bytes()?,"host_retained_bytes":self.retained_bytes(),
            "startup_loader":if self.framework.is_some(){"core_infra::model_load"}else{"full-file pinned fallback"},
            "checkpoint_admitted":self.framework.is_some(),"preparation":self.preparation_report(),
            "declared_startup":self.declared.as_ref().map(|runtime|runtime.borrow().report()),
            "device_admission":{"fresh_components":"native allocator; no static full-execution scratch gate",
                "reload_headroom_bytes":crate::admission::RELOAD_HEADROOM_BYTES,
                "explicit_prefetch":"caller-supplied device budget"},
            "weight_residency":if self.resident{"retain until explicit eviction"}else{"phase scoped"},
            "verified_checkpoints":self.checkpoints.borrow().len(),
            "world_size":if self.peer.is_some(){2}else{1},"devices":self.devices()?,
            "host_asset_bytes":self.assets.bytes(),"verified_host_assets":self.assets.entries(),
            "resident_tokenizers":self.checkpoints.borrow().values().filter(|(_,c)|c.tokenizer_loaded()).count(),
            "tokenizer_initialization_seconds":self.checkpoints.borrow().values().map(|(_,c)|c.tokenizer_initialization_seconds()).sum::<f64>(),
            "transient_pool_bytes":self.device.pool_bytes(),"transient_pool_limit_bytes":crate::cuda::TRANSIENT_POOL_BYTES,
            "reader_policy":native_core_infra::weight_loader::Config::from_env()?,
            "resident_kernel_modules":self.device.kernel_count()}))
    }
    pub fn warm_host(&self,profile:&Path)->Result<Value>{
        self.declared.as_ref().ok_or("host warmup requires a resident declaration")?.borrow().check_host_warm()?;
        let config:Config=serde_json::from_slice(&std::fs::read(profile)?)?;
        if !config.inference || !self.resident{return Err("host warmup requires a resident inference profile".into());}
        let gold=Gold::profile(&config.gold,config.profile_manifests.get("gold").ok_or("profile identity")?)?;
        self.phase("host_prepare");
        let mut startup_checkpoints=Vec::new();
        for (kind,file) in [
            ("transformer","diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
            ("text_encoder","text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
            ("video_vae","vae/ltx-2.5-video-vae-bf16.safetensors"),
            ("audio_vae","vae/ltx-2.5-audio-vae-bf16.safetensors"),
            ("spatial_upsampler","latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")]{
            let checkpoint=self.checkpoint(&gold,&config.weights,kind,file)?;
            if kind!="audio_vae"{startup_checkpoints.push(checkpoint);}
        }
        let checkpoints=Self::framework_groups(&startup_checkpoints);
        let adopted=if let Some(runtime)=&self.declared{runtime.borrow_mut().warm_host(&checkpoints)?}else{false};
        if !adopted && self.retained_bytes()==0 {
            // A completed prefetch has no pending readers. This explicit RAM
            // operation may then materialize its own retained cache.
            for host in crate::startup::prepare_host_world(self.framework.as_ref().ok_or("framework declaration")?,if self.peer.is_some(){2}else{1})?.finish()?{
                crate::startup::install_host(host,&checkpoints)?;
            }
        }
        self.pin_retained_sources()?;
        // Runtime profiles retain only model constants. Captured request
        // activations have no blob in these exported manifests.
        for (kind,path) in [("gold",Some(&config.gold)),("forward",Some(&config.forward)),
            ("text",Some(&config.text)),("upsampler",Some(&config.upsampler)),
            ("encoder",config.encoder.as_ref()),("audio",config.audio.as_ref()),("video",config.video.as_ref())]{
            if let Some(path)=path{
                let mut aux=Gold::profile(path,config.profile_manifests.get(kind).ok_or("profile identity")?)?;
                aux.assets=Some(self.assets.clone());aux.warm_assets()?;
            }
        }
        let secondary=self.peer.as_ref().map(|peer|peer.warm(profile)).transpose()?;
        self.phase("idle");let mut value=self.metadata()?;
        value["device_bytes_uploaded"]=json!(0);
        if self.checkpoint_identity.is_some(){value["preload_inventory"]=self.preload_inventory(&config,&gold,None)?;}
        if let Some(peer)=secondary{
            value["rank_receipts"]=json!([value.clone(),peer]);value["all_ranks_completed"]=json!(true);
        }
        Ok(value)
    }
}
fn positions(frames:usize,height:usize,width:usize,audio:bool,fps:f32,device:&Device)->Result<Tensor>{
    let mut values=Vec::new();
    let shape=if audio{
        for t in 0..frames{for bound in 0..2{values.push(((t+bound)*4).saturating_sub(3) as f32*160.);}}
        vec![1,1,frames,2]
    }else{
        for axis in 0..3{for t in 0..frames{for h in 0..height{for w in 0..width{for bound in 0..2{
            let v=match axis{0=>((t+bound)*8).saturating_sub(7),1=>(h+bound)*32,_=>(w+bound)*32};
            values.push(v as f32);
        }}}}}
        vec![1,3,frames*height*width,2]
    };
    let out=Tensor::contiguous_shape(shape,"torch.float32".into(),device)?;
    upload_into(&out.storage,&values.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>(),device)?;
    let mut times=out.clone();if !audio{times.shape[1]=1;}
    tensor::elementwise(23,&times,None,None,&times,if audio{16000.}else{fps},1.,device.stream)?;
    Ok(out)
}
fn patchify(x:&Tensor,audio:bool,device:&Device)->Result<Tensor>{
    if audio{
        if x.shape.len()!=4 || x.shape[0]!=1 || x.shape[1]!=8 || x.shape[3]!=16{return Err("audio request shape".into());}
        // Flatten C/F after the time permutation. This is a real copy because
        // those dimensions are separated in contiguous NCTF storage.
        let n=x.shape[2];let mut view=x.clone();
        view.shape=vec![1,n,8,16];view.strides=vec![x.strides[0],x.strides[2],x.strides[1],x.strides[3]];
        let mut out=view.contiguous(device)?;out.shape=vec![1,n,128];out.strides=vec![n*128,128,1];Ok(out)
    }else{
        if x.shape.len()!=5 || x.shape[0]!=1 || x.shape[1]!=128{return Err("video request shape".into());}
        let contiguous=x.contiguous(device)?;let n=x.shape[2]*x.shape[3]*x.shape[4];
        let mut out=contiguous;out.shape=vec![1,n,128];out.strides=vec![n*128,1,n];Ok(out)
    }
}
fn component(aux:&Gold,name:&str,activation:&Tensor,weights:&Checkpoint,statistics:Option<&Checkpoint>,
    device:&Device,linear:&Linear)->Result<Tensor>{
    let program:Value=serde_json::from_slice(&std::fs::read(aux.root.join(format!("{name}.graph.json")))?)?;
    let mut replacements=BTreeMap::from([(format!("{name}.args.0"),activation.clone())]);
    for key in graph::external_names(&program)?{
        if let Some(suffix)=key.strip_prefix(&format!("{name}.weights.")){
            let weight_key=if name=="upsampler"{suffix.strip_prefix("upsampler.").ok_or("upsampler parameter prefix")?.to_owned()}
                else if name.starts_with("encoder."){format!("encoder.{suffix}")}else{suffix.to_owned()};
            replacements.insert(key.clone(),text::weight(weights,&weight_key,device)?);
        }else if let Some(suffix)=key.strip_prefix(&format!("{name}.buffers.")){
            let (checkpoint,weight_key)=if let Some(suffix)=suffix.strip_prefix("statistics."){
                (statistics.ok_or("upsampler channel statistics")?,format!("per_channel_statistics.{suffix}"))
            }else{(weights,suffix.into())};
            replacements.insert(key.clone(),text::weight(checkpoint,&weight_key,device)?);
        }
    }
    // Fail closed if composition leaves any request activation unreplaced.
    for key in graph::external_names(&program)?{
        if key.contains(".args.") || key.contains(".kwargs."){
            if !replacements.contains_key(&key){return Err(format!("unreplaced component activation {key}").into());}
        }
    }
    let inputs=graph::load_externals_replacing(aux,&program,device,&replacements)?;
    let (values,report)=graph::execute(&program,inputs,device,linear,false)?;
    if report["executed"]!=true{return Err("component execution stopped".into());}
    let output=program["output_paths"][""].as_str().ok_or("component output")?;
    Ok(values.get(output).ok_or("native component result")?.clone())
}
fn noised(gold:&Gold,stage:usize,modality:usize,source:&Tensor,encoded:Option<&Tensor>,
    generator:&mut CudaGenerator,device:&Device,checks:&mut Vec<Value>)->Result<State>{
    let request=&gold.identity["request"];let audio=modality==1;
    let frames=(request["num_frames"].as_u64().ok_or("frames")? as usize-1)/8+1;
    let scale=if stage==0{64}else{32};
    let height=request["height"].as_u64().ok_or("height")? as usize/scale;
    let width=request["width"].as_u64().ok_or("width")? as usize/scale;
    let source=patchify(source,audio,device)?;let n=source.shape[1];
    let clean=source.contiguous(device)?;
    let mask=filled(vec![1,n,1],"torch.float32",1.,device)?;
    if let Some(encoded)=encoded{
        let tokens=patchify(encoded,false,device)?;let count=tokens.shape[1];
        if audio || count!=height*width{return Err("conditioning token count".into());}
        let mut first=clean.clone();first.shape[1]=count;tensor::copy(&tokens,&first,device.stream)?;
        let mut first=mask.clone();first.shape[1]=count;
        tensor::fill(&first,(1.-request["image_strength"].as_f64().ok_or("image strength")?) as f32,0.,device.stream)?;
    }
    let positions=positions(if audio{n}else{frames},height,width,audio,
        request["frame_rate"].as_f64().ok_or("fps")? as f32,device)?;
    let keyframes=if audio{None}else{
        let mask=filled(vec![1,n,1],"torch.float32",0.,device)?;
        let mut first=mask.clone();first.shape[1]=height*width;tensor::fill(&first,1.,0.,device.stream)?;Some(mask)
    };
    let prefix=format!("noiser.{}",stage*2+modality);
    for (field,value) in [("latent",&source),("clean_latent",&clean),("denoise_mask",&mask),("positions",&positions)]{
        compare(gold,&format!("{prefix}.args.0.{field}"),value,device,checks)?;
    }
    if let Some(value)=&keyframes{compare(gold,&format!("{prefix}.args.0.keyframes_mask"),value,device,checks)?;}
    let draws=gold.metadata["fixtures"].as_array().ok_or("RNG fixtures")?.iter().filter_map(|fixture|{
        let name=fixture["name"].as_str()?;
        (fixture["kind"]=="rng" && gold.metadata[name]["seed"].as_u64()==request["seed"].as_u64()
            && gold.entry(&format!("{name}.out")).ok()?.shape.len()==3).then(||name.to_owned())
    }).collect::<Vec<_>>();
    if gold.verify && draws.len()!=4{return Err("request noiser draw count".into());}
    let draw=draws.get(stage*2+modality).map(String::as_str).unwrap_or("");let metadata=&gold.metadata[draw];
    if gold.verify && metadata["offset_before"].as_u64()!=Some(generator.offset){return Err("request Philox draw order".into());}
    let noise=Tensor::contiguous_shape(source.shape.clone(),"torch.bfloat16".into(),device)?;
    generator.randn(noise.ptr(),noise.elements()?,true,device.stream)?;
    if gold.verify && metadata["offset_after"].as_u64()!=Some(generator.offset){return Err("request Philox offset".into());}
    if gold.verify{compare(gold,&format!("{draw}.out"),&noise,device,checks)?;}
    let source32=cast(&source,"torch.float32",device)?;let noise32=cast(&noise,"torch.float32",device)?;
    let clean32=cast(&clean,"torch.float32",device)?;
    let first=Tensor::contiguous_shape(source.shape.clone(),"torch.float32".into(),device)?;
    tensor::elementwise(37,&source32,Some(&noise32),None,&first,if stage==0{1.}else{0.909375},0.,device.stream)?;
    let second=Tensor::contiguous_shape(source.shape.clone(),"torch.float32".into(),device)?;
    tensor::elementwise(37,&clean32,Some(&first),Some(&mask),&second,0.,1.,device.stream)?;
    let latent=cast(&second,"torch.bfloat16",device)?;
    compare(gold,&format!("{prefix}.out.latent"),&latent,device,checks)?;
    Ok(State{latent,mask,clean,positions:Some(positions),keyframes})
}

pub fn infer(profile:&Path,request:&Path,output:&Path)->Result<Value>{
    infer_on(1,profile,request,output)
}
/// `infer` on `world` GPUs (1, or 2 to run blocks 24-47 on GPU 1).
pub fn infer_on(world:usize,profile:&Path,request:&Path,output:&Path)->Result<Value>{
    let framework=crate::admission::checkpoint(None)?;
    // This owner exits immediately after its one complete request. Keep the
    // in-request GPU/asset reuse, without preparing RAM for a future reload.
    // Session::infer and the native ABI retain their resident cache policy.
    let mut session=Session::open_resident_with_checkpoint_world(framework,world)?;
    session.retain_host_cache=false;
    session.infer(profile,request,output)
}
/// `infer_on` `repeat` times in one resident process: requests after the first are warm. Returns
/// the last request's report with every request's wall seconds and phase start offsets.
pub fn infer_repeat(world:usize,profile:&Path,request:&Path,output:&Path,repeat:usize)->Result<Value>{
    let framework=crate::admission::checkpoint(None)?;
    let mut session=Session::open_resident_with_checkpoint_world(framework,world)?;
    // Weights stay on the GPU between requests; no host copies are prepared for a reload.
    session.retain_host_cache=false;
    let phases=Rc::new(RefCell::new(Vec::<(String,std::time::Instant)>::new()));
    let sink=phases.clone();
    session.set_phase_callback(Box::new(move|name|sink.borrow_mut().push((name.into(),std::time::Instant::now()))));
    let mut runs=Vec::new();let mut report=Value::Null;
    for _ in 0..repeat{
        phases.borrow_mut().clear();
        let started=std::time::Instant::now();
        report=if runs.is_empty(){session.infer(profile,request,output)?}else{
            // Later requests reuse the GPU-resident weights directly (no host-cache preparation).
            let mut config:Config=serde_json::from_slice(&std::fs::read(profile)?)?;
            config.raw_request=Some(request.into());config.output=output.into();
            session.run_warmed(config,None,None)?
        };
        let marks:Vec<Value>=phases.borrow().iter()
            .map(|(name,at)|json!([name,at.duration_since(started).as_secs_f64()])).collect();
        let mut run=json!({"seconds":started.elapsed().as_secs_f64(),"phases":marks});
        if graph::host_profiling(){run["host"]=graph::host_profile_take();}
        eprintln!("{}",json!({"event":"repeat-request","index":runs.len(),"seconds":run["seconds"],"phases":run["phases"],
            "outputs":report["outputs"],"decoded_outputs":report["decoded"]["video"]["outputs"],
            "waveform":report["decoded"]["audio"]["output"]}));
        runs.push(run);
    }
    report["requests"]=json!(runs);Ok(report)
}
impl Session {
pub fn infer(&self,profile:&Path,request:&Path,output:&Path)->Result<Value>{
    self.infer_prepared(profile,request,output,None,None)
}
pub fn infer_prepared(&self,profile:&Path,request:&Path,output:&Path,
    generation:Option<u64>,gate:Option<native_core_infra::phase_gate::Pin>)->Result<Value>{
    // A declared session's GPU 1 warms the profile before copying its blocks.
    let peer_profile=self.peer.as_ref().and(self.declared.as_ref()).map(|_|profile);
    let mut config:Config=serde_json::from_slice(&std::fs::read(profile)?)?;
    if !config.inference{return Err("expected an exported native profile".into());}
    if let Some(generation)=generation{
        if generation!=self.generation.get() || !self.prepared_profile.borrow().as_ref()
            .is_some_and(|(path,digest)|path==profile && *digest==sha(&std::fs::read(profile).unwrap_or_default())){
            return Err("request weight lease differs from the prepared profile/generation".into());
        }
    }else if self.declared.as_ref().is_some_and(|runtime|runtime.borrow().needs_prefetch())
        || (self.declared.is_none() && self.resident && self.pending.borrow().is_none()){
        self.prepare_weights(profile,self.device.available_bytes()?.saturating_sub(crate::admission::RELOAD_HEADROOM_BYTES),false)?;
    }
    config.raw_request=Some(request.into());config.output=output.into();self.run_warmed(config,gate,peer_profile)
}
pub fn run(&self,config:Config)->Result<Value>{self.run_with_gate(config,None)}
pub fn run_with_gate(&self,config:Config,gate:Option<native_core_infra::phase_gate::Pin>)->Result<Value>{
    self.run_warmed(config,gate,None)
}
/// `peer_profile`: the profile GPU 1 warms, which then copies its blocks from GPU 0.
fn run_warmed(&self,mut config:Config,gate:Option<native_core_infra::phase_gate::Pin>,
    peer_profile:Option<&Path>)->Result<Value>{
    let peer_start=self.peer.as_ref().map(|peer|peer.completed_forwards()).unwrap_or(0);
    let first_dit=native_core_infra::execution_start::Capture::begin();
    let asset_uploads_before=self.device.asset_upload_counts();
    let open=|kind:&str,root:&Path|->Result<Gold>{let mut value=if config.inference{
        Gold::profile(root,config.profile_manifests.get(kind).ok_or("profile manifest checksum")?)?
    }else{Gold::open(root)?};
        if self.resident && config.inference{value.assets=Some(self.assets.clone());}Ok(value)};
    let profiles=native_core_infra::startup_trace::Span::new("startup_request","profile manifests",0);
    let mut gold=open("gold",&config.gold)?;let mut forward=open("forward",&config.forward)?;
    let mut text_gold=open("text",&config.text)?;let mut upsampler=open("upsampler",&config.upsampler)?;
    let mut encoder=config.encoder.as_ref().map(|root|open("encoder",root)).transpose()?;
    drop(profiles);
    if config.inference{
        let path=config.raw_request.as_ref().ok_or("raw request JSON is required")?;
        let input:Value=serde_json::from_slice(&std::fs::read(path)?)?;
        for (key,value) in input.as_object().ok_or("raw request object")?{
            match key.as_str(){
                "prompt"=>{if value.as_str().is_none(){return Err("prompt must be a string".into());}gold.identity["request"][key]=value.clone();}
                "seed"=>{if value.as_u64().is_none(){return Err("seed must be an unsigned integer".into());}gold.identity["request"][key]=value.clone();}
                "image"=>{let image=PathBuf::from(value.as_str().ok_or("image filename")?);
                    config.raw_image=Some(if image.is_absolute(){image}else{path.parent().ok_or("request directory")?.join(image)});}
                "height"|"width"|"num_frames"|"frame_rate"|"image_strength"=>{
                    if value!=&gold.identity["request"][key]{return Err(format!("request parameter {key} differs from the admitted geometry/branch").into());}
                }
                _=>return Err(format!("unported request parameter {key}").into()),
            }
        }
        if input.get("prompt").is_none() || input.get("seed").is_none(){return Err("raw request requires prompt and seed".into());}
        if encoder.is_some()!=config.raw_image.is_some(){return Err("raw image presence must match the admitted T2V/I2V branch".into());}
        for aux in [&mut forward,&mut text_gold,&mut upsampler].into_iter().chain(encoder.iter_mut()){
            aux.identity["request"]=gold.identity["request"].clone();
        }
    }
    for aux in [&forward,&text_gold,&upsampler]{
        if aux.identity["source_gold"].as_str()!=gold.root.to_str(){return Err("request profile source differs".into());}
    }
    let device=&self.device;let linear=&self.linear;
    let transformer=self.checkpoint(&gold,&config.weights,"transformer","diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors")?;
    let text_encoder=self.checkpoint(&gold,&config.weights,"text_encoder","text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")?;
    let video_vae=self.checkpoint(&gold,&config.weights,"video_vae","vae/ltx-2.5-video-vae-bf16.safetensors")?;
    let spatial=self.checkpoint(&gold,&config.weights,"spatial_upsampler","latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")?;
    let checkpoints=Self::framework_groups(&[transformer.clone(),text_encoder.clone(),video_vae.clone(),spatial.clone()]);
    if let Some(runtime)=&self.declared {runtime.borrow_mut().request(text_gold.root.clone(),checkpoints.clone())?;}
    if self.framework.is_some() && self.pending.borrow().is_some(){self.prepare_text_backend(&text_gold)?;}
    let mut rest=if let Some(mut pending)=self.pending.borrow_mut().take(){
        crate::startup::install(unsafe{pending.finish_first(device.stream)}?,&checkpoints,device)?;
        Some(pending.spawn_rest()?)
    }else{None};
    self.finish_cpu_preparation(&text_encoder)?;
    self.phase("text");
    let text_use=self.use_component("text")?;let mut connector_use=None;
    let (contexts,report)=text::contexts_ready(&text_gold,&text_encoder,&transformer,&device,&linear,||{
        connector_use=self.use_component("connectors")?;
        if let Some(rest)=&mut rest{crate::startup::install(rest.next_stage()?.0,&checkpoints,device)?;}
        Ok(())
    })?;
    drop(connector_use);drop(text_use);
    if report["passed"]!=true{return Ok(report);}
    // Resident weight lifetime belongs to the session owner. Keep
    // Gemma as well as the transformer for the next same-model request;
    // Session::evict releases both partitions on an actual model switch.
    if !self.resident{device.clear_weight_cache()?;}
    if let Some(rest)=rest{
        for (loaded,_) in rest.join().map_err(|_|"LTX stage thread panicked")?? {
            crate::startup::install(loaded,&checkpoints,device)?;
        }
    }
    let _dit_use=self.use_component("dit")?;
    // GPU 1 warms only once GPU 0's reads are done, with GPU 0's parsed gold
    // profile: parsing or reading files during those reads slowed them.
    let replica=match (peer_profile,&self.peer){
        (Some(profile),Some(peer))=>{
            let warm=peer.begin_warm(profile,&gold)?;let replica=peer.begin_replicate(&transformer,device)?;
            warm.finish()?;Some(replica.finish()?)
        }
        _=>None,
    };
    let _vae_use=self.use_component("vae")?;
    let _upscale_use=self.use_component("upscale")?;
    self.phase("sampling");
    let mut checks=report["checks"].as_array().ok_or("text checks")?.clone();
    let contexts=[contexts[0].clone(),contexts[1].clone()];
    let mut host_preparation=None;
    let request=&gold.identity["request"];let frames=request["num_frames"].as_u64().ok_or("frames")? as usize;
    let fps=request["frame_rate"].as_f64().ok_or("fps")?;
    let height=request["height"].as_u64().ok_or("height")? as usize;
    let width=request["width"].as_u64().ok_or("width")? as usize;
    let mut sources=vec![filled(vec![1,128,(frames-1)/8+1,height/64,width/64],"torch.bfloat16",0.,&device)?,
        filled(vec![1,8,(frames as f64/fps*25.).round_ties_even() as usize,16],"torch.bfloat16",0.,&device)?];
    let mut generator=CudaGenerator::new(request["seed"].as_u64().ok_or("seed")?)?;
    let image=if let Some(path)=&config.raw_image{
        if gold.verify && sha(&std::fs::read(path)?)!=request["image_sha256"].as_str().ok_or("raw image identity")?{return Err("raw image checksum differs".into());}
        Some(crate::media::decode(path,encoder.as_ref().ok_or("image encoder profile")?.metadata["image.crf"].as_i64().ok_or("image CRF")? as i32)?)
    }else{None};
    if request.get("image_file").is_some() && encoder.is_none(){return Err("I2V requires an encoder profile".into());}
    let mut sampling_steps=Vec::new();
    for stage in 0..2{
        let encoded=if let Some(encoder)=&encoder{
            // The current I2V boundary is the encoder's preprocessed image.
            // Raw image decoding is validated separately before widening it.
            let name=format!("encoder.{stage}");let key=format!("{name}.args.0");let entry=encoder.entry(&key)?;
            let input=if let Some(image)=&image{
                let scale=if stage==0{2}else{1};crate::media::prepare(image,height/scale,width/scale,&device)?
            }else{
                let input=Tensor::contiguous_shape(entry.shape.clone(),entry.dtype.clone(),&device)?;
                let data=encoder.asset(&key)?;upload_into(&input.storage,&data,&device)?;input
            };
            compare(encoder,&key,&input,&device,&mut checks)?;
            let value=component(encoder,&name,&input,&video_vae,None,&device,&linear)?;
            compare(encoder,&format!("{name}.out"),&value,&device,&mut checks)?;Some(value)
        }else{None};
        let initial=[noised(&gold,stage,0,&sources[0],encoded.as_ref(),&mut generator,&device,&mut checks)?,
            noised(&gold,stage,1,&sources[1],None,&mut generator,&device,&mut checks)?];
        eprintln!("{}",json!({"event":"native-request-front-passed","stage":stage}));
        if config.front_only{return Ok(json!({"passed":true,"rung":"request-front","checks":checks}));}
        let outcome=sampling::run_parallel(&gold,&forward,&transformer,stage,false,Some(&initial),Some(&contexts),&device,&linear,self.peer.as_ref())?;
        checks.extend(outcome.report["checks"].as_array().ok_or("sampling checks")?.iter().cloned());
        if outcome.report["passed"]!=true{return Ok(json!({"passed":false,"rung":"request-latents","checks":checks}));}
        sampling_steps.push(outcome.report["steps"].as_u64().ok_or("completed sampling steps")? as usize);
        if stage==0 && self.retain_host_cache && self.retained_bytes()==0 {
            if let Some(declaration)=&self.framework{host_preparation=Some(crate::startup::prepare_host_world(declaration,
                if self.peer.is_some(){2}else{1})?);}
        }
        sources=outcome.states;
        if stage==0{
            sources[0]=component(&upsampler,"upsampler",&sources[0],&spatial,Some(&video_vae),&device,&linear)?;
            compare(&gold,"upsampler.0.out",&sources[0],&device,&mut checks)?;
        }
    }
    std::fs::create_dir_all(&config.output)?;let mut outputs=Vec::new();
    for (name,value) in ["video","audio"].into_iter().zip(&sources){
        let raw=value.download(&device)?;let path=config.output.join(format!("{name}.latent.raw"));
        std::fs::write(&path,&raw)?;outputs.push(json!({"path":path,"shape":value.shape,"dtype":value.dtype,"sha256":sha(&raw)}));
    }
    if !self.resident{
        device.clear_weight_cache()?;
        if let Some(peer)=&self.peer{peer.evict()?;}
    }
    if let Some(preparation)=host_preparation {
        self.phase("host_cache_finalize");
        for host in preparation.finish()?{crate::startup::install_host(host,&checkpoints)?;}
        eprintln!("{}",json!({"event":"retained-host-cache-ready","bytes":self.retained_bytes()}));
    }
    // Sampling has drained before registering retained sources or retiring weights.
    self.pin_retained_sources()?;
    self.phase("decode_boundary");
    let decode_handoff=if let Some(gate)=gate{
        Some(native_core_infra::phase_gate::acknowledge_with_retirement(&gate,"decode",&[height,width,frames],||{
            // Both sampling stages and every encoder/upsampler use have ended.
            // Only the owning CUDA thread mutates its caches. Decoder weights,
            // latents and retained host snapshots remain live.
            let before=device.cached_weight_tensors();
            let free_before=device.available_bytes()?;
            let peer_before=self.peer.as_ref().map(|peer|peer.status()).transpose()?;
            device.evict_checkpoint(&transformer.path)?;
            device.evict_checkpoint(&text_encoder.path)?;
            device.evict_checkpoint(&spatial.path)?;
            let peer_after=self.peer.as_ref().map(|peer|peer.evict()).transpose()?;
            self.prepared_profile.borrow_mut().take();
            self.prepared_receipt.borrow_mut().take();
            let mut devices=vec![json!({"device_uuid":device.uuid()?,"free_before_bytes":free_before,
                "free_after_bytes":device.available_bytes()?})];
            if let (Some(before),Some(after))=(peer_before,peer_after){devices.push(json!({
                "device_uuid":before["device_uuid"],"free_before_bytes":before["device_free_bytes"],
                "free_after_bytes":after["device_free_bytes"]}));}
            let receipt=json!({"retired":true,"primary_tensors_before":before,
                "primary_tensors_after":device.cached_weight_tensors(),"ranks":if self.peer.is_some(){2}else{1},
                "decoder_weights_preserved":true,"host_weights_preserved":true,"devices":devices});
            // Publish before sending the retirement receipt: the framework can
            // submit a foreign loader as soon as it receives that receipt.
            self.phase(native_core_infra::phase_gate::DECODE_WEIGHTS_RETIRED);
            Ok(receipt)
        })?)
    }else{None};
    self.phase(if decode_handoff.as_ref().is_some_and(|v|v["retirement"]["retired"]==true){
        native_core_infra::phase_gate::DECODE_WEIGHTS_RETIRED
    }else{"decode"});
    let mut decoded=json!({});let mut waveform=None;let mut frame_chunks=None;let mut audio_rate=0;
    for (kind,path) in [("audio",&config.audio),("video",&config.video)]{
        if let Some(path)=path{
            let mut aux=if config.inference{Gold::profile(path,config.profile_manifests.get(kind).ok_or("decoder profile checksum")?)?}else{Gold::open(path)?};
            if self.resident && config.inference{aux.assets=Some(self.assets.clone());}
            if aux.identity["source_gold"].as_str()!=gold.root.to_str(){return Err("decoder profile source differs".into());}
            if kind=="audio"{
                self.checkpoint(&gold,&config.weights,"audio_vae","vae/ltx-2.5-audio-vae-bf16.safetensors")?;
                let (sound,report)=crate::decode::audio(&aux,&sources[1],&device,&linear)?;
                audio_rate=aux.metadata["sampling_rate"].as_u64().ok_or("audio sampling rate")? as usize;
                decoded["audio"]=json!({"report":report,"output":crate::decode::write_raw(&config.output,"audio.f32.raw",&sound,&device)?});
                waveform=Some(sound);
            }else{
                let (frames,report)=crate::decode::video(&aux,&sources[0],request["seed"].as_u64().ok_or("seed")?,&device,&linear)?;
                // Large decoder temporaries and its convolution workspace
                // have been released. Hashing/writing/encoding remain.
                self.phase("decode_tail");
                let mut chunks=Vec::new();
                for (index,frame) in frames.iter().enumerate(){chunks.push(crate::decode::write_raw(&config.output,&format!("video.frames.{index}.raw"),frame,&device)?);}
                decoded["video"]=json!({"report":report,"outputs":chunks});
                frame_chunks=Some(frames);
            }
        }
    }
    if let (Some(sound),Some(frames))=(&waveform,&frame_chunks){
        decoded["mp4"]=crate::media::encode(&config.output.join("output.mp4"),frames,sound,fps as usize,audio_rate,&device)?;
    }
    drop(_upscale_use);drop(_vae_use);drop(_dit_use);
    let forbidden=std::fs::read_to_string("/proc/self/maps")?.lines().filter(|line|
        ["libpython","libtorch","libc10","libaten"].iter().any(|name|line.contains(name))).map(str::to_owned).collect::<Vec<_>>();
    if !forbidden.is_empty(){return Err("native runtime loaded a forbidden inference library".into());}
    self.phase("complete");
    let expected_forwards:usize=sampling_steps.iter().sum();
    let parallel=if let Some(peer)=&self.peer{
        let forwards=peer.completed_forwards()-peer_start;
        if forwards==0 && crate::forward::graphs_enabled(){
            // Sampling replayed CUDA graphs of all 48 blocks on GPU 0 (see sampling::run_parallel).
            json!({"world_size":2,"mode":"gpu0-cuda-graphs","gpu1_dit_replica":replica,"devices":self.devices()?,
                "forwards_per_rank":[expected_forwards,0],"block_evaluations_per_rank":[expected_forwards*48,0]})
        }else{
        if forwards!=expected_forwards{return Err("both LTX partitions must complete every configured DiT forward".into());}
        json!({"world_size":2,"mode":"pipeline-model-parallel","boundary_transfer":"peer copy of each tensor's allocation",
            "gpu1_dit_replica":replica,"devices":self.devices()?,"all_ranks_completed":true,"forwards_per_rank":forwards,
            "blocks_per_rank":[24,24],"block_evaluations_per_rank":[forwards*24,forwards*24]})
        }
    }else{json!({"world_size":1,"devices":self.devices()?,"forwards_per_rank":expected_forwards,
        "all_ranks_completed":true,"block_evaluations_per_rank":[expected_forwards*48]})};
    Ok(json!({"passed":true,"rung":if config.inference{"native-inference"}else{"request-latents"},"checks":checks,"outputs":outputs,"decoded":decoded,
        "parallel":parallel,"sampling_steps_per_stage":sampling_steps,"decode_handoff":decode_handoff,
        "first_dit_start":first_dit.report()?,
        "declared_startup":self.declared.as_ref().map(|runtime|runtime.borrow().report()),
        "host_retained_bytes":self.retained_bytes(),"host_pinned_bytes":self.pinned_host_bytes()?,
        "host_cache_policy":if self.retain_host_cache{"resident-reuse"}else{"request-lifetime"},
        "model_constant_uploads":{"tensors":device.asset_upload_counts().0-asset_uploads_before.0,
            "payload_bytes":device.asset_upload_counts().1-asset_uploads_before.1},
        "forbidden_mapped_libraries":forbidden,
        "input_boundary":if image.is_some(){"raw prompt and raw image"}else if encoder.is_some(){"raw prompt and preprocessed image"}else{"raw prompt"}}))
}
}
