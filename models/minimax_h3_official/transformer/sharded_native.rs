// Included by native.rs. Shards remain resident; a request may retain all
// reconstructed layers through every denoising step, then release them together.
use std::cell::{Cell,RefCell};
use crate::weight_shards::GatherMode;
unsafe extern "C" {
    fn cudaEventCreateWithFlags(event:*mut *mut c_void,flags:u32)->c_int;
    fn cudaEventRecord(event:*mut c_void,stream:Stream)->c_int;
    fn cudaEventElapsedTime(ms:*mut f32,start:*mut c_void,end:*mut c_void)->c_int;
    fn cudaEventDestroy(event:*mut c_void)->c_int;
    fn cudaStreamWaitEvent(stream:Stream,event:*mut c_void,flags:u32)->c_int;
    fn h3_comm_duplicate(comm:*mut c_void,out:*mut *mut c_void)->c_int;
    fn h3_comm_stream(comm:*mut c_void)->Stream;
    fn h3_comm_allgather(comm:*mut c_void,send:*const c_void,send_elements:usize,
        recv:*mut c_void,recv_elements:usize,dtype:c_int)->c_int;
}
struct GatherEvent(*mut c_void);
impl GatherEvent {
    fn new(timing:bool)->Result<Self>{let mut e=Self(ptr::null_mut());cuda(unsafe{cudaEventCreateWithFlags(&mut e.0,if timing{0}else{2})})?;Ok(e)}
    fn record(&self,stream:Stream)->Result<()>{cuda(unsafe{cudaEventRecord(self.0,stream)})}
    fn wait(&self,stream:Stream)->Result<()>{cuda(unsafe{cudaStreamWaitEvent(stream,self.0,0)})}
    fn seconds_to(&self,end:&Self)->Result<f64>{let mut ms=0.0;cuda(unsafe{cudaEventElapsedTime(&mut ms,self.0,end.0)})?;Ok(ms as f64/1000.0)}
}
impl Drop for GatherEvent{fn drop(&mut self){if !self.0.is_null(){unsafe{cudaEventDestroy(self.0);}}}}
struct GatherSample {start:GatherEvent,end:GatherEvent,wait_start:GatherEvent,wait_end:GatherEvent,consumed:Cell<bool>}
impl GatherSample {
    fn new()->Result<Self>{Ok(Self{start:GatherEvent::new(true)?,end:GatherEvent::new(true)?,
        wait_start:GatherEvent::new(true)?,wait_end:GatherEvent::new(true)?,consumed:Cell::new(false)})}
}
struct CachedLayer {buffer:Buffer,ready:GatherEvent,loaded:Cell<bool>,consumed:Cell<bool>,sample:Cell<Option<usize>>}
struct RequestWeights {layers:HashMap<String,CachedLayer>,bytes:usize,_arena:Arena}
impl RequestWeights {
    fn new(manifest:&crate::weight_shards::Manifest)->Result<Self>{
        let (bytes,layout)=manifest.cache_layout()?;
        // Independent arena: cudaFree at request end, never retained by RequestPool.
        let mut arena=Arena::new()?;let full=arena.alloc(bytes)?;let mut layers=HashMap::new();
        for (name,(offset,len)) in layout {
            layers.insert(name,CachedLayer{buffer:full.slice(offset,len)?,ready:GatherEvent::new(false)?,
                loaded:Cell::new(false),consumed:Cell::new(false),sample:Cell::new(None)});
        }
        Ok(Self{layers,bytes,_arena:arena})
    }
}
struct GatherWorkspace {
    buffer:Buffer,active:RefCell<Option<(String,String)>>,group:std::rc::Rc<CpGroup>,mode:GatherMode,
    samples:Vec<GatherSample>,sample_count:Cell<usize>,_arena:Arena,
}
impl GatherWorkspace {
    fn new(bytes:usize,parent:std::rc::Rc<CpGroup>)->Result<std::rc::Rc<Self>> {
        let mode=GatherMode::parse(&std::env::var("H3_WEIGHT_GATHER_MODE").unwrap_or_else(|_|"p2p".into()))?;
        let group=if mode.overlap(){let mut handle=ptr::null_mut();cp_check(unsafe{h3_comm_duplicate(parent.handle,&mut handle)})?;
            std::rc::Rc::new(CpGroup{handle,rank:parent.rank,world:parent.world,direct:std::cell::RefCell::new(None)})}else{parent};
        let mut arena=Arena::new()?;let buffer=arena.alloc(bytes)?;
        // No event creation in timed layer loops. Bound telemetry for full base schedules.
        let samples=(0..4096).map(|_|GatherSample::new()).collect::<Result<Vec<_>>>()?;
        Ok(std::rc::Rc::new(Self{buffer,active:RefCell::new(None),group,mode,samples,sample_count:Cell::new(0),_arena:arena}))
    }
}
impl Drop for GatherWorkspace {fn drop(&mut self){let _=cp_check(unsafe{h3_comm_synchronize(self.group.handle)});}}
#[derive(Default)]
struct GatherStats {calls:usize,cache_hits:usize,prefetches:usize,prefetch_hits:usize,step_start:usize,step_calls:Vec<usize>,
    cache_peak_bytes:usize,cache_allocate_seconds:f64,cache_release_seconds:f64,
    materialized_bytes:u64,send_bytes:u64,receive_bytes:u64,gpu_seconds:f64,host_seconds:f64,wait_seconds:f64}
/// The communicator a shard owner uses: ready, or forming on a helper thread.
enum GroupSource {Ready(std::rc::Rc<CpGroup>),Pending(PendingGroup)}
impl GroupSource {
    fn world_rank(&self)->(usize,usize){match self{Self::Ready(g)=>(g.world,g.rank),Self::Pending(p)=>(p.world,p.rank)}}
    fn resolve(self)->Result<std::rc::Rc<CpGroup>>{match self{Self::Ready(g)=>Ok(g),Self::Pending(p)=>p.join()}}
}
fn shard_manifest(settings:&Value)->Result<(PathBuf,crate::weight_shards::Manifest,String)> {
    let root=PathBuf::from(settings["root"].as_str().ok_or("weight shard root missing")?).canonicalize()?;
    let bytes=crate::startup_trace::read(root.join("manifest.json"))?;
    let digest=crate::integrity::digest(&bytes);
    if Some(digest.as_str())!=settings["manifest_sha256"].as_str(){return Err("weight shard manifest changed".into());}
    Ok((root,crate::weight_shards::Manifest::parse(&bytes)?,digest))
}
/// This rank's shard bytes on bounded parallel readers, started before CUDA and the communicator.
/// With slices, job i fills `pieces[i]` = (group, offset in this rank's shard); without, job i is
/// group i's shard file. `ShardedWeights::load` admits the manifest again before any upload.
pub(crate) struct ShardReaders {pool:crate::weight_loader::ReaderPool,config:crate::weight_loader::Config,
    manifest_sha256:String,pieces:Option<Vec<(usize,usize)>>}
impl ShardReaders {
    pub(crate) fn start(model:&Value,weights_root:&Path)->Result<Option<Self>> {
        let Some(config)=crate::weight_loader::Config::from_env()? else {return Ok(None)};
        let _trace=crate::startup_trace::Span::new("dit_shard_reader_start","",0);
        let settings=&model["weight_sharding"];
        let (root,manifest,digest)=shard_manifest(settings)?;
        let rank=cp_env_usize("H3_CP_RANK",0)?;
        if rank>=manifest.world_size{return Err("CP rank outside weight shard world".into());}
        let (jobs,pieces)=if let Some(wanted)=settings.get("slices_sha256") {
            let raw=crate::startup_trace::read(root.join("slices.json"))?;
            if Some(crate::integrity::digest(&raw).as_str())!=wanted.as_str(){return Err("DiT shard slices changed".into());}
            let slices:crate::weight_shards::Slices=serde_json::from_slice(&raw)?;
            let (mut jobs,mut pieces)=(vec![],vec![]);
            for piece in slices.rank(&manifest,&digest,rank)? {
                jobs.push(crate::weight_loader::Job::slice(weights_root,&piece.file,piece.offset,piece.bytes,
                    piece.sha256.clone(),piece.file_bytes)?);
                pieces.push((piece.group,piece.dest));
            }
            (jobs,Some(pieces))
        }else{
            (manifest.groups.iter().map(|g|{let s=&g.ranks[rank];
                crate::weight_loader::Job::new(&root,&s.file,s.bytes,Some(s.sha256.clone()))})
                .collect::<std::result::Result<Vec<_>,String>>()?,None)
        };
        Ok(Some(Self{pool:crate::weight_loader::ReaderPool::start(jobs,config)?,config,manifest_sha256:digest,pieces}))
    }
}
struct ShardedWeights {
    manifest:crate::weight_shards::Manifest,manifest_sha256:String,
    shards:HashMap<String,Buffer>,key_groups:HashMap<String,String>,offsets:HashMap<String,usize>,
    workspace:std::rc::Rc<GatherWorkspace>,group:std::rc::Rc<CpGroup>,cache:RefCell<Option<RequestWeights>>,
    stats:RefCell<GatherStats>,reconstruction:RefCell<Value>,ready:Cell<bool>,
    // The request cache outlives the reply; freed before the next command.
    pending_release:Cell<bool>,
    transaction:RefCell<crate::request_transaction::Sequence>,
}
impl ShardedWeights {
    fn verify_resident_payload(&self,stream:Stream)->Result<Value> {
        // Sharded serving does not need the streaming loader enabled. The
        // audit only needs a bounded D2H buffer, not a change in load policy.
        let config=crate::weight_loader::Config::from_env()?.unwrap_or(crate::weight_loader::Config {
            workers:1,chunk_bytes:16<<20,pinned_slots:1,budget_bytes:64<<20,
        });
        let verify=crate::weight_loader::device::Upload::new(stream,config)?;
        let mut bytes=0usize;
        for layer in &self.manifest.groups {
            let spec=&layer.ranks[self.group.rank];
            let shard=self.shards.get(&layer.name).ok_or("resident weight shard missing")?;
            if shard.bytes!=spec.bytes{return Err("resident weight shard size changed".into());}
            unsafe{verify.verify(shard.ptr,shard.bytes,&spec.sha256)}?;
            bytes+=shard.bytes;
        }
        Ok(json!({"expected_resident_shard_sha_verified":true,"rank":self.group.rank,
            "world_size":self.group.world,"groups":self.manifest.groups.len(),"bytes":bytes,
            "manifest_sha256":self.manifest_sha256}))
    }
    fn load(arena:&mut Arena,model:&Value,model_bytes:&[u8],group:GroupSource,
            shared:Option<std::rc::Rc<GatherWorkspace>>,capacity:usize,files:&mut Vec<String>,
            readers:Option<ShardReaders>)
            ->Result<(Self,HashMap<String,Tensor>)> {
        let (world,rank)=group.world_rank();
        if !crate::weight_shards::SUPPORTED_WORLDS.contains(&world){return Err("weight sharding requires CP world=2, 4 or 8".into());}
        let (root,manifest,digest)=shard_manifest(&model["weight_sharding"])?;
        if manifest.world_size!=world{return Err("weight shard world differs from communicator".into());}
        let expected=crate::weight_shards::expected_tensors(model)?;
        let recipe=crate::recipe::Recipe::from_model(model)?;
        manifest.validate(&expected,recipe.workflow,&crate::integrity::base_identity(model,model_bytes)?)?;
        // The readers verify every job as it lands (light guard or full SHA-256), before this returns.
        let mut streamed=if let Some(mut readers)=readers {
            if readers.manifest_sha256!=digest{return Err("shard readers started from another manifest".into());}
            let _trace=crate::startup_trace::Span::new("dit_shard_upload",if readers.pieces.is_some(){"slices"}else{"files"},0);
            let sizes=manifest.groups.iter().map(|g|g.ranks[rank].bytes).collect::<Vec<_>>();
            let buffers=arena.alloc_model_tensors(&sizes)?;
            let mut upload=crate::weight_loader::device::Upload::new(arena.stream,readers.config)?;
            match &readers.pieces {
                Some(pieces)=>{
                    // The slices cover every tensor byte; the alignment gaps between them are zero.
                    for buffer in &buffers {cuda(unsafe{cudaMemsetAsync(buffer.ptr,0,buffer.bytes,arena.stream)})?;}
                    unsafe{readers.pool.consume_upload(&mut upload,|i,offset|{let (group,dest)=pieces[i];
                        buffers[group].ptr.cast::<u8>().add(dest+offset).cast()})}?;
                },
                None=>unsafe{readers.pool.consume_upload(&mut upload,|i,offset|buffers[i].ptr.cast::<u8>().add(offset).cast())}?,
            }
            upload.finish()?;
            Some(buffers.into_iter())
        }else{None};
        // A communicator that formed during the upload is joined only now, before the gather
        // workspace and the first collective need it.
        let group=group.resolve()?;
        let workspace=if let Some(value)=shared {
            if value.buffer.bytes<manifest.gather_bytes(){return Err("shared layer gather capacity too small".into());}value
        }else{GatherWorkspace::new(capacity.max(manifest.gather_bytes()),group.clone())?};
        let mut shards=HashMap::new();let mut views=HashMap::new();let mut key_groups=HashMap::new();
        let mut offsets=HashMap::new();
        for layer in &manifest.groups {
            let spec=&layer.ranks[group.rank];
            let shard=if let Some(buffers)=&mut streamed {
                files.push(format!("{}#rank-{}",layer.name,group.rank));buffers.next().ok_or("streamed weight shard missing")?
            }else{
                let path=root.join(&spec.file).canonicalize()?;
                if !path.starts_with(&root){return Err("rank shard escapes package root".into());}
                let data=crate::startup_trace::read(&path)?;
                if data.len()!=spec.bytes || crate::integrity::digest(&data)!=spec.sha256{return Err("rank shard bytes changed".into());}
                files.push(path.to_string_lossy().into_owned());arena.upload(&data)?
            };
            shards.insert(layer.name.clone(),shard);
            for entry in &layer.entries {
                let view=workspace.buffer.slice(entry.offset,entry.tensor.bytes)?;
                for name in &entry.names {
                    views.insert(name.clone(),Tensor{spec:expected[name].clone(),buffer:view});
                    key_groups.insert(name.clone(),layer.name.clone());offsets.insert(name.clone(),entry.offset);
                }
            }
        }
        let weights=views.into_iter().filter_map(|(key,value)|key.strip_prefix("weights/").map(|name|(name.to_owned(),value))).collect();
        arena.sync()?;
        let result=Self{manifest,manifest_sha256:digest,shards,key_groups,workspace,group,
            stats:RefCell::new(GatherStats::default()),reconstruction:RefCell::new(json!({"performed":false})),ready:Cell::new(true),offsets,cache:RefCell::new(None),
            transaction:RefCell::new(crate::request_transaction::Sequence::default()),pending_release:Cell::new(false)};
        if crate::debug_audit::enabled(){result.audit_reconstruction(arena)?;}
        Ok((result,weights))
    }

    fn enqueue(&self,name:&str,recv:Buffer,compute:Stream,synchronous:bool,prefetch:bool)->Result<usize> {
        let ws=&self.workspace;let n=ws.sample_count.get();let e=ws.samples.get(n).ok_or("weight gather telemetry capacity exceeded")?;
        let stream=if ws.mode.overlap(){unsafe{h3_comm_stream(ws.group.handle)}}else{ws.group.bind(compute)?;compute};
        let started=Instant::now();e.start.record(stream)?;let send=self.shards[name];
        if ws.mode==GatherMode::P2p {let rows=vec![send.bytes;self.group.world];
            cp_check(unsafe{h3_comm_allgatherv(ws.group.handle,send.ptr,send.bytes,recv.ptr,recv.bytes,rows.as_ptr(),1,0)})?;
        }else{cp_check(unsafe{h3_comm_allgather(ws.group.handle,send.ptr,send.bytes,recv.ptr,recv.bytes,0)})?;}
        e.end.record(stream)?;e.consumed.set(false);
        if synchronous{cp_check(unsafe{h3_comm_synchronize(ws.group.handle)})?;}
        ws.sample_count.set(n+1);let mut stats=self.stats.borrow_mut();stats.calls+=1;stats.prefetches+=usize::from(prefetch);
        stats.materialized_bytes+=recv.bytes as u64;let traffic=(send.bytes*(self.group.world-1)) as u64;
        stats.send_bytes+=traffic;stats.receive_bytes+=traffic;stats.host_seconds+=started.elapsed().as_secs_f64();Ok(n)
    }
    fn enqueue_cached(&self,name:&str,cache:&RequestWeights,stream:Stream,prefetch:bool)->Result<()> {
        let layer=cache.layers.get(name).ok_or("missing request cache layer")?;
        if layer.loaded.get(){return Ok(());}
        let index=self.enqueue(name,layer.buffer,stream,!self.workspace.mode.overlap(),prefetch)?;
        let producer=unsafe{h3_comm_stream(self.workspace.group.handle)};
        layer.ready.record(producer)?;layer.sample.set(Some(index));layer.loaded.set(true);Ok(())
    }
    fn materialize(&self,name:&str,stream:Stream)->Result<()> {
        if !self.ready.get(){return Err("weight shard owner is not ready".into());}
        let ws=&self.workspace;let identity=(self.manifest_sha256.clone(),name.to_owned());
        if let Some(cache)=self.cache.borrow().as_ref() {
            let layer=cache.layers.get(name).ok_or("missing request layer")?;
            if layer.loaded.get(){self.stats.borrow_mut().cache_hits+=1;
                if !layer.consumed.get(){self.stats.borrow_mut().prefetch_hits+=1;}}
            self.enqueue_cached(name,cache,stream,false)?;
            if !layer.consumed.get(){
                let e=&ws.samples[layer.sample.get().ok_or("cached layer without gather")?];
                e.wait_start.record(stream)?;layer.ready.wait(stream)?;e.wait_end.record(stream)?;
                e.consumed.set(true);layer.consumed.set(true);
            }
        }else{
            if ws.active.borrow().as_ref()==Some(&identity){self.stats.borrow_mut().cache_hits+=1;return Ok(());}
            let layer=self.manifest.groups.iter().find(|g|g.name==name).ok_or("missing weight group")?;
            // Startup audits and non-cached baselines use the original single buffer.
            // Previous consumer completes before any overwrite, even on a separate producer.
            self.group.bind(stream)?;
            self.enqueue(name,ws.buffer.prefix(layer.bytes)?,stream,true,false)?;
        }
        *ws.active.borrow_mut()=Some(identity);Ok(())
    }
    fn prefetch_block(&self,layer:usize,stream:Stream)->Result<()> {
        if self.workspace.mode.overlap() && layer+1<50 {
            if let Some(cache)=self.cache.borrow().as_ref(){self.enqueue_cached(&format!("transformer_blocks.{}",layer+1),cache,stream,true)?;}
        }Ok(())
    }
    fn resolve(&self,key:&str,original:&Tensor)->Result<Tensor> {
        self.require_key(key)?;let mut tensor=original.clone();
        if let Some(cache)=self.cache.borrow().as_ref(){
            let layer=&cache.layers[&self.key_groups[key]];
            if !layer.consumed.get(){return Err("cached weights used before readiness dependency".into());}
            tensor.buffer=layer.buffer.slice(self.offsets[key],original.spec.bytes)?;
        }Ok(tensor)
    }
    fn require_key(&self,key:&str)->Result<()> {
        let group=self.key_groups.get(key).ok_or("unknown sharded weight view")?;
        if self.workspace.active.borrow().as_ref()!=Some(&(self.manifest_sha256.clone(),group.clone())) {
            return Err(format!("weight view used before its layer was gathered: {key}").into());
        }Ok(())
    }
    fn drain(&self,stream:Stream)->Result<()> {
        let ws=&self.workspace;
        cp_check(unsafe{h3_comm_synchronize(ws.group.handle)})?;
        cp_check(unsafe{h3_comm_synchronize(self.group.handle)})?;
        cuda(unsafe{cudaStreamSynchronize(stream)})?;
        let mut stats=self.stats.borrow_mut();
        for e in ws.samples.iter().take(ws.sample_count.get()){
            stats.gpu_seconds+=e.start.seconds_to(&e.end)?;
            if e.consumed.get(){stats.wait_seconds+=e.wait_start.seconds_to(&e.wait_end)?;}
        }
        ws.sample_count.set(0);
        Ok(())
    }
    fn finish_request(&self,stream:Stream)->Result<()> {
        self.drain(stream)?;
        let ws=&self.workspace;let mut stats=self.stats.borrow_mut();
        if self.cache.borrow().is_some(){
            let started=Instant::now();drop(self.cache.borrow_mut().take());stats.cache_release_seconds+=started.elapsed().as_secs_f64();
            // Only resident shards and the startup reconstruction buffer remain.
            *ws.active.borrow_mut()=None;
        }Ok(())
    }
    /// Complete the request's gathers now; free its weight cache after the reply (`release_deferred`).
    fn defer_release(&self,stream:Stream)->Result<()> {
        self.drain(stream)?;self.pending_release.set(true);Ok(())
    }
    /// Seconds spent freeing a deferred request cache, or None when nothing was pending.
    fn release_deferred(&self,stream:Stream)->Result<Option<f64>> {
        if !self.pending_release.replace(false){return Ok(None);}
        let started=Instant::now();self.finish_request(stream)?;Ok(Some(started.elapsed().as_secs_f64()))
    }
    fn audit_reconstruction(&self,arena:&Arena)->Result<()> {
        let mut records=Vec::new();let mut tensors=0usize;
        for layer in &self.manifest.groups {
            self.materialize(&layer.name,arena.stream)?;
            let bytes=arena.download(self.workspace.buffer.prefix(layer.bytes)?)?;
            let digest=crate::integrity::digest(&bytes);
            if digest!=layer.sha256{return Err("all-gather reconstructed different layer bytes".into());}
            for entry in &layer.entries {
                let digest=crate::integrity::digest(&bytes[entry.offset..entry.offset+entry.tensor.bytes]);
                if Some(digest.as_str())!=entry.tensor.sha256.as_deref(){return Err("reconstructed tensor differs from original parameter".into());}
                tensors+=entry.names.len();
            }
            records.push(json!({"group":layer.name,"sha256":digest,"bytes":layer.bytes,"passed":true}));
        }
        self.finish_request(arena.stream)?;
        *self.reconstruction.borrow_mut()=json!({"performed":true,"passed":true,"groups":records,
            "logical_tensors_checked":tensors,"reference":"original complete tensor SHA256; zero-padded rank-file concatenation SHA256"});Ok(())
    }
    fn begin_request(&self,phase:Option<&crate::request_transaction::Phase>)->Result<()> {
        // Normally released right after the previous reply; never carried into a new request.
        self.release_deferred(unsafe{h3_comm_stream(self.group.handle)})?;
        self.transaction.borrow().check(&self.manifest.workflow,phase)?;
        if phase.is_some_and(|p|p.index>0) {
            // A request cache carries every gathered layer into the next phase. Per-layer gathers
            // (p2p/allgather) keep nothing between phases, so the previous one only has to be drained.
            let carried=!self.workspace.mode.cache() || self.cache.borrow().as_ref().is_some_and(|c|
                c.layers.values().all(|l|l.loaded.get() && l.consumed.get()));
            if self.workspace.sample_count.get()!=0 || !carried {
                return Err("previous phase did not leave its weight gathers drained".into());
            }
            self.transaction.borrow_mut().advance(&self.manifest.workflow,phase)?;
            return Ok(());
        }
        if self.cache.borrow().is_some() || self.workspace.sample_count.get()!=0{return Err("preceding request did not drain weight streams/cache".into());}
        *self.stats.borrow_mut()=GatherStats::default();
        if self.workspace.mode.cache(){
            let started=Instant::now();let cache=RequestWeights::new(&self.manifest)?;
            let mut stats=self.stats.borrow_mut();stats.cache_peak_bytes=cache.bytes;stats.cache_allocate_seconds=started.elapsed().as_secs_f64();
            *self.cache.borrow_mut()=Some(cache);*self.workspace.active.borrow_mut()=None;
        }
        self.transaction.borrow_mut().advance(&self.manifest.workflow,phase)?;
        Ok(())
    }
    fn begin_step(&self){let mut stats=self.stats.borrow_mut();stats.step_start=stats.calls;}
    fn end_step(&self,_phase_step:usize)->Result<()> {
        let mut stats=self.stats.borrow_mut();let calls=stats.calls-stats.step_start;
        let step=stats.step_calls.len(); // global evaluation count across this outer request
        if self.workspace.mode.cache() && (step==0 && calls!=self.manifest.groups.len() || step>0 && calls!=0){
            return Err("request cache must gather all layers exactly once in step zero".into());
        }
        stats.step_calls.push(calls);Ok(())
    }
    fn report(&self)->Value {
        let s=self.stats.borrow();let ws=&self.workspace;
        json!({"enabled":true,"world_size":self.group.world,"rank":self.group.rank,"model_load_count":1,
            "manifest_sha256":self.manifest_sha256,"workflow":self.manifest.workflow,
            "resident_shard_bytes":self.shards.values().map(|b|b.bytes).sum::<usize>(),
            "full_packed_model_bytes":self.manifest.groups.iter().map(|g|g.bytes).sum::<usize>(),
            "shared_gather_buffer_bytes":ws.buffer.bytes,"shared_gather_buffer_pointer":ws.buffer.ptr as usize,
            "gather_mode":ws.mode.name(),"independent_communicator":ws.mode.overlap(),
            "request_cache_peak_bytes":s.cache_peak_bytes,"request_cache_current_bytes":self.cache.borrow().as_ref().map_or(0,|c|c.bytes),
            "cache_transaction_open":!self.transaction.borrow().is_idle(),
            "request_cache_allocate_seconds":s.cache_allocate_seconds,"request_cache_release_seconds":s.cache_release_seconds,
            "groups":self.manifest.groups.len(),"request_model_file_reads":0,"strategy":ws.mode.name(),
            "communication":{"calls":s.calls,"cache_hits":s.cache_hits,"prefetches":s.prefetches,"prefetch_hits":s.prefetch_hits,
                "step_gather_calls":s.step_calls,"materialized_bytes":s.materialized_bytes,
                "send_bytes_per_rank":s.send_bytes,"receive_bytes_per_rank":s.receive_bytes,
                "gpu_seconds":s.gpu_seconds,"host_seconds":s.host_seconds,"compute_wait_gpu_seconds":s.wait_seconds,
                "scope":"gather GPU events; compute wait events measure exposed readiness waits; host is enqueue span (includes completion only in synchronous modes)"},
            "reconstruction":self.reconstruction.borrow().clone()})
    }
}
impl Drop for ShardedWeights {
    fn drop(&mut self){let _=self.finish_request(unsafe{h3_comm_stream(self.group.handle)});}
}
