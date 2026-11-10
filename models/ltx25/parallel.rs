//! Two physical GPUs run the DiT's 48 blocks sequence parallel on every forward (see `sp`): each
//! takes half of the video tokens through all blocks. Each thread owns its CUDA context, allocator,
//! handles and kernel cache. Tensors cross as exact peer copies that preserve logical values and
//! strides; model arithmetic is exactly the original block graph, with no cross-device reductions.
//! At cold start GPU 1 copies the blocks from GPU 0's DiT over the peer link instead of reading
//! them again.
use crate::{backbone,cuda::{Buffer,Device},gold::{self,Gold},linear::Linear,
    request::Config,tensor::{self,Tensor},weights::Checkpoint,Result};
use native_core_infra::device_context::copy_peer;
use serde_json::{json,Value};
use std::{cell::{Cell,RefCell},collections::BTreeMap,path::{Path,PathBuf},rc::Rc,sync::mpsc,thread,time::Instant};

/// The DiT tensors GPU 1 holds: every transformer block (the input projections and output heads
/// run on GPU 0 only).
pub fn peer_partition(key:&str)->bool{
    key.strip_prefix("model.diffusion_model.transformer_blocks.")
        .and_then(|suffix|suffix.split('.').next()).and_then(|block|block.parse::<usize>().ok())
        .is_some_and(|block|block<48)
}

/// A tensor on the other GPU, named by address for a peer copy.
struct PeerTensor {address:u64,bytes:usize,shape:Vec<usize>,strides:Vec<usize>,dtype:String,offset:usize,reference_offset:usize}
impl PeerTensor {
    /// The entire backing allocation. Recorded view operations can address a
    /// sibling region before/after the logical tensor's origin.
    fn whole(value:&Tensor)->Self{Self{address:value.storage.dev(),bytes:value.storage.bytes(),shape:value.shape.clone(),
        strides:value.strides.clone(),dtype:value.dtype.clone(),offset:value.offset,reference_offset:value.reference_offset}}
    /// Only the bytes of a contiguous weight, which shares its allocation with its whole stage.
    fn weight(value:&Tensor)->Result<Self>{
        if value.strides!=tensor::contiguous_strides(&value.shape) || value.reference_offset!=0{
            return Err("LTX weight is not a contiguous view".into());
        }
        let bytes=value.elements()?.checked_mul(tensor::dtype(&value.dtype)?.1).ok_or("weight extent")?;
        Ok(Self{address:value.ptr(),bytes,shape:value.shape.clone(),strides:value.strides.clone(),
            dtype:value.dtype.clone(),offset:0,reference_offset:0})
    }
    /// A copy into a new allocation on `device`, from the private context `source`.
    fn copy(&self,source:usize,device:&Device)->Result<Tensor>{
        // A stream-ordered transient: reused only by work queued after this copy on `device`.
        let storage=device.transient(self.bytes)?;
        unsafe{copy_peer(storage.dev(),device.context_identity()?,self.address,source,self.bytes,device.stream)}?;
        Ok(Tensor{storage:Rc::new(storage),shape:self.shape.clone(),strides:self.strides.clone(),
            dtype:self.dtype.clone(),offset:self.offset,reference_offset:self.reference_offset})
    }
}
type Reply<T>=mpsc::Sender<std::result::Result<T,String>>;
enum Work {
    Warm(PathBuf,Option<gold::Shared>,Reply<Value>),Prefetch(PathBuf,usize,bool,Reply<Value>),Identity(String,Reply<Value>),
    Replicate(usize,Vec<(String,PeerTensor)>,Reply<Value>),
    /// Stage, step, request, GPU 0's context, an event after GPU 0 wrote the inputs, the inputs.
    Forward(usize,usize,Value,usize,usize,BTreeMap<String,PeerTensor>,crate::sp::Exchange,Reply<(BTreeMap<String,PeerTensor>,Value)>),
    Evict(Reply<Value>),Status(Reply<Value>),Stop,
}
struct Half {
    identity:Option<String>,
    framework:Option<native_core_infra::checkpoint::Declaration>,
    checkpoints:BTreeMap<PathBuf,Rc<Checkpoint>>,
    profiles:BTreeMap<PathBuf,(Gold,[Value;2],Rc<Checkpoint>)>,
    current:Option<PathBuf>,budget:native_core_infra::pinned_weights::HostBudget,
    assets:Rc<native_core_infra::immutable_assets::ImmutableAssets>,
    linear:Linear,device:Device,blocks:usize,
    /// Whether warming reads a host copy of the partition for later reloads.
    retain_host:bool,
    /// GPU 0's context once linked, and the last outputs until GPU 0 has copied them.
    source:Option<usize>,returned:Option<BTreeMap<String,Tensor>>,
}
impl Half {
    fn open(framework:Option<native_core_infra::checkpoint::Declaration>,retain_host:bool)->Result<Self>{
        let mut device=Device::open_private(1)?;
        let assets=Rc::new(native_core_infra::immutable_assets::ImmutableAssets::new(2usize<<30,4096));
        device.host_assets=Some(assets.clone());let linear=Linear::new(&device)?;
        Ok(Self{identity:None,framework,checkpoints:BTreeMap::new(),profiles:BTreeMap::new(),current:None,
            budget:native_core_infra::pinned_weights::HostBudget::new(48usize<<30),
            assets,linear,device,blocks:0,retain_host,source:None,returned:None})
    }
    fn link(&mut self,source:usize)->Result<()>{
        if self.source!=Some(source){self.device.enable_peer(source)?;self.source=Some(source);}Ok(())
    }
    fn status(&self)->Result<Value>{Ok(json!({"device_index":1,"device_uuid":self.device.uuid()?,
        "device_free_bytes":self.device.available_bytes()?,
        "cached_weight_tensors":self.device.cached_weight_tensors(),
        "owner_pid":std::process::id(),"owner_instance":native_core_infra::process_identity::current(),
        "host_pinned_bytes":self.budget.used()+self.checkpoints.values().flat_map(|c|c.retained_hosts())
            .map(|host|host.pinned_bytes()).sum::<usize>(),
        "startup_loader":if self.framework.is_some(){"core_infra::model_load"}else{"full-file pinned fallback"},
        "first_block":0,"end_block":48,
        "completed_block_evaluations":self.blocks,"resident_kernel_modules":self.device.kernel_count()}))}
    /// `shared`: GPU 0's parsed gold profile, which this GPU then need not parse again.
    fn warm(&mut self,path:&Path,shared:Option<gold::Shared>)->Result<Value>{
        if !self.profiles.contains_key(path){
            let config:Config=serde_json::from_slice(&std::fs::read(path)?)?;
            if !config.inference{return Err("dual LTX requires an exported raw-request inference profile".into());}
            let mut gold=match shared.filter(|shared|shared.root()==config.gold){
                Some(shared)=>shared.gold(None),
                None=>Gold::profile(&config.gold,config.profile_manifests.get("gold").ok_or("gold pin")?)?,
            };
            gold.assets=Some(self.assets.clone());gold.warm_assets()?;
            let checkpoint_path=config.weights.join("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors");
            let checkpoint=if let Some(value)=self.checkpoints.get(&checkpoint_path){value.clone()}else{
                let mut value=Checkpoint::open(&checkpoint_path)?;
                let expected=gold.identity["weights"]["transformer"]["sha256"].as_str().ok_or("transformer pin")?;
                if let Some(declaration)=&self.framework{
                    value.admit("dit",expected,std::sync::Arc::new(declaration.manifest.clone()),&declaration.roots)?;
                    // Without a host copy, the blocks come from GPU 0 or the admitted files.
                    if self.retain_host{
                        let requests=declaration.manifest.tensors.iter().filter(|(name,entry)|
                            name.strip_prefix("dit.").is_some_and(peer_partition) &&
                            matches!(entry.dtype,native_core_infra::checkpoint::Dtype::BF16|native_core_infra::checkpoint::Dtype::F32))
                            .map(|(name,entry)|native_core_infra::model_load::Request{name:name.clone(),dtype:entry.dtype}).collect();
                        let mut options=native_core_infra::model_load::Options::for_declaration(declaration)?;
                        options.retained_budget=48usize<<30;
                        for host in native_core_infra::model_load::HostPreparation::start(declaration.manifest.clone(),
                            declaration.roots.clone(),vec![requests],options)?.finish()?{
                            for name in host.names(){value.retain(name.strip_prefix("dit.").ok_or("partition tensor prefix")?,name,host.clone());}
                        }
                    }
                }else{value.pin(expected,&self.budget)?;}
                let value=Rc::new(value);self.checkpoints.insert(checkpoint_path,value.clone());value
            };
            let templates=[serde_json::from_slice(&std::fs::read(gold.root.join("block.0.0.0.graph.json"))?)?,
                           serde_json::from_slice(&std::fs::read(gold.root.join("block.1.0.0.graph.json"))?)?];
            self.profiles.insert(path.into(),(gold,templates,checkpoint));
        }
        self.current=Some(path.into());let mut value=self.status()?;
        if self.identity.is_some(){value["preload_inventory"]=self.inventory(path,None)?;}
        Ok(value)
    }
    fn inventory(&self,path:&Path,uploaded:Option<&[String]>)->Result<Value>{
        let checkpoint=&self.profiles[path].2;
        let config:Config=serde_json::from_slice(&std::fs::read(path)?)?;
        let(mut count,mut bytes,mut resident_count,mut resident_bytes,mut copied_count,mut copied_bytes)=(0,0,0,0,0,0);
        for(key,info)in checkpoint.header.as_object().ok_or("checkpoint header")?{
            if !peer_partition(key)||!matches!(info["dtype"].as_str(),Some("BF16"|"F32")){continue;}
            let offsets=info["data_offsets"].as_array().ok_or("weight offsets")?;
            let size=offsets[1].as_u64().and_then(|n|n.checked_sub(offsets[0].as_u64()?)).ok_or("weight extent")? as usize;
            count+=1;bytes+=size;
            if let Some(value)=self.device.cached_weight(&format!("{}\0{key}",checkpoint.path.display())){
                if value.elements()?.checked_mul(crate::tensor::dtype(&value.dtype)?.1)!=Some(size){return Err("LTX peer resident tensor extent differs".into());}
                resident_count+=1;resident_bytes+=size;
            }
            if uploaded.is_some_and(|keys|keys.contains(key)){copied_count+=1;copied_bytes+=size;}
        }
        if count==0||checkpoint.retained_bytes()==0{return Err("LTX peer host inventory missing".into());}
        let mut row=json!({"name":"dit","priority":if config.encoder.is_some(){2}else{1},
            "manifest_sha256":self.identity.as_ref().ok_or("LTX peer checkpoint identity missing")?,
            "tensor_count":count,"payload_bytes":bytes,"verified_tensor_count":count,"verified_payload_bytes":bytes,
            "host_ready":true,"device_uuid":self.device.uuid()?,"reader_policy":native_core_infra::weight_loader::Config::from_env()?});
        if uploaded.is_some(){
            row["resident_tensor_count"]=json!(resident_count);row["resident_payload_bytes"]=json!(resident_bytes);
            row["uploaded_tensor_count"]=json!(copied_count);row["uploaded_payload_bytes"]=json!(copied_bytes);
            row["reused_tensor_count"]=json!(resident_count-copied_count);row["reused_payload_bytes"]=json!(resident_bytes-copied_bytes);
            row["copy_complete"]=json!(resident_count==count&&resident_bytes==bytes);
        }
        Ok(json!({"schema":"native-preload-inventory-v1","components":[row],
            "dispatch_order":if copied_bytes>0{vec!["dit"]}else{vec![]}}))
    }
    fn prefetch(&mut self,path:&Path,budget:usize,copy:bool)->Result<Value>{
        self.warm(path,None)?;
        let checkpoint=&self.profiles[path].2;
        let keys=checkpoint.header.as_object().ok_or("checkpoint header")?.iter()
            .filter(|(key,info)|peer_partition(key) && matches!(info["dtype"].as_str(),Some("BF16"|"F32")))
            .filter(|(key,_)|copy&&self.device.cached_weight(&format!("{}\0{key}",checkpoint.path.display())).is_none())
            .map(|(key,_)|key.clone()).collect::<Vec<_>>();
        let bytes=keys.iter().try_fold(0usize,|sum,key|->Result<usize>{
            let offsets=checkpoint.header[key]["data_offsets"].as_array().ok_or("weight offsets")?;
            let bytes=offsets[1].as_u64().and_then(|end|end.checked_sub(offsets[0].as_u64()?)).ok_or("weight extent")?;
            sum.checked_add(bytes.try_into()?).ok_or_else(||"weight sum overflow".into())
        })?;
        if bytes>budget{return Err("secondary LTX partition exceeds its device copy budget".into());}
        let start=native_core_infra::execution_start::monotonic_ns();
        checkpoint.prefetch_weights(&keys,&self.device)?;self.device.sync()?;
        let end=native_core_infra::execution_start::monotonic_ns();
        let mut status=self.status()?;status["device_bytes_uploaded"]=json!(bytes);
        status["upload_started_monotonic_ns"]=json!(start);status["upload_completed_monotonic_ns"]=json!(end);
        if self.identity.is_some(){status["preload_inventory"]=self.inventory(path,Some(&keys))?;}
        status["gpu_prepare_complete"]=json!(copy);status["copy_complete"]=json!(true);
        Ok(status)
    }
    /// Copies GPU 0's resident blocks into one allocation of this device's cache.
    fn replicate(&mut self,source:usize,tensors:Vec<(String,PeerTensor)>)->Result<Value>{
        self.link(source)?;
        let path=self.current.as_ref().ok_or("dual LTX profile not warmed")?;
        let checkpoint=self.profiles.get(path).ok_or("dual profile missing")?.2.clone();
        let started=Instant::now();let aligned=|bytes:usize|bytes.div_ceil(256)*256;
        let total=tensors.iter().map(|(_,value)|aligned(value.bytes)).sum::<usize>();
        let _span=native_core_infra::startup_trace::Span::new("model_replicate","LTX DiT blocks 0-47",total);
        let storage=Rc::new(Buffer::new(total)?);let target=self.device.context_identity()?;let mut offset=0;
        for (key,value) in &tensors{
            unsafe{copy_peer(storage.at(offset),target,value.address,source,value.bytes,self.device.stream)}?;
            self.device.cache_weight(format!("{}\0{key}",checkpoint.path.display()),&Tensor{storage:storage.clone(),offset,
                shape:value.shape.clone(),strides:value.strides.clone(),dtype:value.dtype.clone(),reference_offset:0});
            offset+=aligned(value.bytes);
        }
        self.device.sync()?;
        Ok(json!({"device_uuid":self.device.uuid()?,"tensors":tensors.len(),"bytes":total,
            "seconds":started.elapsed().as_secs_f64()}))
    }
    /// GPU 1's token range of a sequence-parallel forward: all 48 blocks of the local program.
    #[allow(clippy::too_many_arguments)]
    fn forward(&mut self,stage:usize,step:usize,request:Value,source:usize,written:usize,inputs:BTreeMap<String,PeerTensor>,
        exchange:crate::sp::Exchange)->Result<(BTreeMap<String,PeerTensor>,Value)>{
        self.link(source)?;
        let path=self.current.as_ref().ok_or("dual LTX profile not warmed")?;
        let (gold,templates,checkpoint)=self.profiles.get_mut(path).ok_or("dual profile missing")?;
        gold.identity["request"]=request;
        let local=crate::sp::local(&gold.root,stage,&templates[stage],crate::sp::split(exchange.total(),exchange.rank()).1)?;
        // Copy the inputs once GPU 0 has written them; GPU 0 keeps them until this reply arrives.
        crate::cuda::wait_event(self.device.stream,written)?;
        let inputs=inputs.into_iter().map(|(key,value)|Ok((key,value.copy(source,&self.device)?)))
            .collect::<Result<BTreeMap<_,_>>>()?;
        let (values,report)=backbone::execute_range_with(gold,&local.graph,checkpoint,stage,step,inputs,
            &self.device,&self.linear,0..48,Some(&exchange))?;
        if report["passed"]!=true{return Err("secondary LTX token range failed".into());}
        let result=values.into_iter().filter(|(key,_)|key.ends_with(".video.x") || key.ends_with(".audio.x"))
            .collect::<BTreeMap<_,_>>();
        if result.len()!=2{return Err("secondary LTX output boundary incomplete".into());}
        let outputs=result.iter().map(|(key,value)|(key.clone(),PeerTensor::whole(value))).collect();
        self.returned=Some(result);self.blocks+=48;Ok((outputs,report))
    }
}

pub struct ParallelBlocks {sender:mpsc::Sender<Work>,thread:Option<thread::JoinHandle<()>>,forwards:Cell<usize>,
    /// GPU 1's context once it has started, and whether GPU 0 may reach it.
    boot:RefCell<Option<mpsc::Receiver<std::result::Result<usize,String>>>>,context:Cell<usize>,linked:Cell<bool>}
/// The peer owns the CUDA context and all upload leases. Even if the root
/// upload fails, drain its terminal reply before the session can evict weights
/// or tear down either context.
pub(crate) struct Pending {reply:Option<mpsc::Receiver<std::result::Result<Value,String>>>}
impl Pending {
    pub(crate) fn finish(mut self)->Result<Value>{
        self.reply.take().ok_or("peer reply already consumed")?
            .recv().map_err(|_|"LTX GPU 1 stopped before replying")?.map_err(Into::into)
    }
}
impl Drop for Pending {
    fn drop(&mut self){if let Some(reply)=self.reply.take(){let _=reply.recv();}}
}
/// GPU 1's copy of GPU 0's blocks in flight. Dropping it waits for the
/// copy, which reads `_sources`.
pub(crate) struct Replica {pending:Pending,_sources:Vec<Tensor>}
impl Replica {pub(crate) fn finish(self)->Result<Value>{self.pending.finish()}}
impl ParallelBlocks {
    pub fn open()->Result<Self>{
        Self::open_with_checkpoint(None)
    }
    pub fn open_with_checkpoint(framework:Option<native_core_infra::checkpoint::Declaration>)->Result<Self>{
        let value=Self::start(framework,true)?;value.booted()?;Ok(value)
    }
    /// Starts GPU 1 on its own thread without waiting for it. `retain_host`
    /// keeps a host copy of its blocks for reloads after an eviction.
    pub fn start(framework:Option<native_core_infra::checkpoint::Declaration>,retain_host:bool)->Result<Self>{
        let (sender,receiver)=mpsc::channel();let (boot,ready)=mpsc::channel();
        let thread=thread::Builder::new().name("native-ltx25-gpu1".into()).spawn(move||{
            let mut half=match Half::open(framework,retain_host){Ok(value)=>value,Err(error)=>{
                eprintln!("LTX GPU 1: {error}");let _=boot.send(Err(error.to_string()));return;}};
            if boot.send(half.device.context_identity().map_err(|error|error.to_string())).is_err(){return;}
            macro_rules! reply {($sender:expr,$result:expr)=>{{let value=$result.map_err(|error|error.to_string());let _=$sender.send(value);}}}
            while let Ok(work)=receiver.recv(){
                // GPU 0 has copied the last outputs before it sends anything else.
                half.returned=None;
                match work{
                Work::Warm(path,shared,sender)=>reply!(sender,half.warm(&path,shared)),
                Work::Prefetch(path,budget,copy,sender)=>reply!(sender,half.prefetch(&path,budget,copy)),
                Work::Identity(identity,sender)=>{half.identity=Some(identity);reply!(sender,half.status());},
                Work::Replicate(source,tensors,sender)=>reply!(sender,half.replicate(source,tensors)),
                Work::Forward(stage,step,request,source,written,inputs,exchange,sender)=>
                    reply!(sender,half.forward(stage,step,request,source,written,inputs,exchange)),
                Work::Evict(sender)=>reply!(sender,half.device.clear_weight_cache().and_then(|_|half.status())),
                Work::Status(sender)=>reply!(sender,half.status()),Work::Stop=>break,
            }}
        })?;
        Ok(Self{sender,thread:Some(thread),forwards:Cell::new(0),boot:RefCell::new(Some(ready)),context:Cell::new(0),linked:Cell::new(false)})
    }
    fn booted(&self)->Result<usize>{
        if let Some(boot)=self.boot.borrow_mut().take(){self.context.set(boot.recv().map_err(|e|e.to_string())??);}
        match self.context.get(){0=>Err("LTX GPU 1 failed to start".into()),context=>Ok(context)}
    }
    /// Lets GPU 0, current on the calling thread, reach GPU 1's allocations.
    fn link(&self,device:&Device)->Result<usize>{
        let context=self.booted()?;
        if !self.linked.get(){device.enable_peer(context)?;self.linked.set(true);}Ok(context)
    }
    /// Queues `work` behind GPU 1's start without waiting for it.
    fn send<T>(&self,work:impl FnOnce(Reply<T>)->Work)->Result<mpsc::Receiver<std::result::Result<T,String>>>{
        let (sender,receiver)=mpsc::channel();
        if self.sender.send(work(sender)).is_err(){self.booted()?;return Err("LTX GPU 1 stopped".into());}
        Ok(receiver)
    }
    fn request<T>(&self,work:impl FnOnce(Reply<T>)->Work)->Result<T>{
        match self.send(work)?.recv(){Ok(value)=>value.map_err(Into::into),Err(_)=>{self.booted()?;Err("LTX GPU 1 stopped".into())}}
    }
    pub fn warm(&self,path:&Path)->Result<Value>{self.request(|reply|Work::Warm(path.into(),None,reply))}
    /// Warms `path` with GPU 0's parsed gold profile.
    pub(crate) fn begin_warm(&self,path:&Path,gold:&Gold)->Result<Pending>{
        Ok(Pending{reply:Some(self.send(|reply|Work::Warm(path.into(),Some(gold.share()),reply))?)})
    }
    /// Starts GPU 1's copy of the blocks from GPU 0's resident `checkpoint` tensors.
    pub(crate) fn begin_replicate(&self,checkpoint:&Checkpoint,device:&Device)->Result<Replica>{
        self.link(device)?;device.sync()?;
        let mut tensors=Vec::new();let mut sources=Vec::new();
        for (key,info) in checkpoint.header.as_object().ok_or("checkpoint header")?{
            if !peer_partition(key) || !matches!(info["dtype"].as_str(),Some("BF16"|"F32")){continue;}
            let value=device.cached_weight(&format!("{}\0{key}",checkpoint.path.display())).ok_or("GPU 0 DiT is not resident")?;
            tensors.push((key.clone(),PeerTensor::weight(&value)?));sources.push(value);
        }
        let source=device.context_identity()?;
        Ok(Replica{pending:Pending{reply:Some(self.send(|reply|Work::Replicate(source,tensors,reply))?)},_sources:sources})
    }
    pub(crate) fn set_checkpoint_identity(&self,identity:String)->Result<()>{self.request(|reply|Work::Identity(identity,reply))?;Ok(())}
    pub(crate) fn begin_prefetch(&self,path:&Path,budget:usize)->Result<Pending>{
        self.begin_prefetch_components(path,budget,true)
    }
    pub(crate) fn begin_prefetch_components(&self,path:&Path,budget:usize,copy:bool)->Result<Pending>{
        Ok(Pending{reply:Some(self.send(|reply|Work::Prefetch(path.into(),budget,copy,reply))?)})
    }
    pub fn prefetch(&self,path:&Path,budget:usize)->Result<Value>{self.begin_prefetch(path,budget)?.finish()}
    pub fn evict(&self)->Result<Value>{self.request(Work::Evict)}
    pub fn status(&self)->Result<Value>{self.request(Work::Status)}
    pub fn completed_forwards(&self)->usize{self.forwards.get()}
    /// The 48 blocks of one forward, sequence parallel: GPU 0 runs its token range of `inputs` here
    /// while GPU 1 runs the other, and the block outputs are assembled on GPU 0. `template` is the
    /// stage's block program at all tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self,gold:&Gold,template:&Value,checkpoint:&Checkpoint,stage:usize,step:usize,
        mut inputs:BTreeMap<String,Tensor>,device:&Device,linear:&Linear)->Result<(BTreeMap<String,Tensor>,Value)>{
        if gold.verify{return Err("sequence-parallel forwards are not reference-verified per tensor".into());}
        let context=self.link(device)?;
        let prefix=format!("block.{stage}.0.0");
        let (video,audio)=(format!("{prefix}.kwargs.video.x"),format!("{prefix}.kwargs.audio.x"));
        let total=inputs.get(&video).ok_or("block inputs have no video tokens")?.shape[1];
        let ranges=[crate::sp::split(total,0),crate::sp::split(total,1)];
        let locals=[crate::sp::local(&gold.root,stage,template,ranges[0].1)?,crate::sp::local(&gold.root,stage,template,ranges[1].1)?];
        let mut own=BTreeMap::new();let mut theirs=BTreeMap::new();let mut held=Vec::new();
        for (key,value) in &inputs{
            match locals[0].axes.get(key){
                Some(&axis)=>{
                    own.insert(key.clone(),crate::sp::narrow(value,axis,ranges[0],&locals[0].graph["externals"][key],device)?);
                    let other=crate::sp::narrow(value,axis,ranges[1],&locals[1].graph["externals"][key],device)?;
                    theirs.insert(key.clone(),PeerTensor::whole(&other));held.push(other);
                }
                None=>{own.insert(key.clone(),value.clone());theirs.insert(key.clone(),PeerTensor::whole(value));}
            }
        }
        // GPU 1 reads its inputs once GPU 0 has written them.
        let written=crate::cuda::Event::record(device.stream)?;
        let source=device.context_identity()?;
        let (exchange,peer_exchange)=crate::sp::Exchange::pair(total,[source,context]);
        let request=gold.identity["request"].clone();
        let reply=self.send(|reply|Work::Forward(stage,step,request,source,written.handle(),theirs,peer_exchange,reply))?;
        let own_result=backbone::execute_range_with(gold,&locals[0].graph,checkpoint,stage,step,own,device,linear,0..48,Some(&exchange));
        // If this half failed, drop its end first: GPU 1's next exchange then fails instead of waiting.
        let exchange=if own_result.is_ok(){Some(exchange)}else{drop(exchange);None};
        // Drain GPU 1's reply even if this half failed.
        let peer_result:Result<(BTreeMap<String,PeerTensor>,Value)>=match reply.recv(){
            Ok(value)=>value.map_err(Into::into),Err(_)=>{self.booted()?;Err("LTX GPU 1 stopped".into())}};
        let (values,report)=own_result?;let (outputs,peer_report)=peer_result?;
        if report["passed"]!=true{return Err("primary LTX token range failed".into());}
        // Video tokens in order: GPU 0's range, then GPU 1's. Audio is identical on both GPUs.
        let own_video=values.get(&video).ok_or("missing primary video output")?;
        let peer_video=outputs.get(&video).ok_or("missing secondary video output")?.copy(context,device)?;
        let mut shape=own_video.shape.clone();shape[1]=total;
        let full=Tensor::contiguous_shape(shape,own_video.dtype.clone(),device)?;
        for (part,(start,count)) in [(own_video,ranges[0]),(&peer_video,ranges[1])]{
            if part.shape[1]!=count{return Err("token range output extent differs".into());}
            let mut rows=full.clone();rows.shape[1]=count;rows.offset+=start*full.strides[1]*tensor::dtype(&full.dtype)?.1;
            tensor::copy(part,&rows,device.stream)?;
        }
        inputs.insert(video,full);
        inputs.insert(audio.clone(),values.get(&audio).ok_or("missing primary audio output")?.clone());
        // GPU 1 frees its outputs with the next command; its input copies were done before it replied.
        device.sync()?;drop(held);drop(exchange);drop(written);
        self.forwards.set(self.forwards.get()+1);
        let mut checks=report["checks"].as_array().cloned().unwrap_or_default();
        checks.extend(peer_report["checks"].as_array().cloned().unwrap_or_default());
        Ok((inputs,serde_json::json!({"passed":true,"rung":"backbone","checks":checks,"sequence_parallel":{
            "tokens":total,"ranges":ranges,"device_uuid":report["device_uuid"],"peer_device_uuid":peer_report["device_uuid"]}})))
    }
}

#[cfg(test)]
mod prefetch_tests {
    use super::*;
    #[test]
    fn abandoned_root_preparation_joins_peer_before_releasing_its_scope(){
        let (sender,receiver)=mpsc::channel();
        let finished=std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let peer_finished=finished.clone();
        let peer=thread::spawn(move||{
            peer_finished.store(true,std::sync::atomic::Ordering::Release);
            sender.send(Ok(json!({"copy_complete":true}))).unwrap();
        });
        drop(Pending{reply:Some(receiver)});
        assert!(finished.load(std::sync::atomic::Ordering::Acquire));peer.join().unwrap();
        let (sender,receiver)=mpsc::channel();sender.send(Err("peer copy failed".into())).unwrap();
        assert!(Pending{reply:Some(receiver)}.finish().unwrap_err().to_string().contains("peer copy failed"));
    }
}
impl Drop for ParallelBlocks {
    fn drop(&mut self){let _=self.sender.send(Work::Stop);if let Some(thread)=self.thread.take(){let _=thread.join();}}
}
