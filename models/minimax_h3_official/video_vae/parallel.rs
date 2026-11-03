// Each thread owns one CUDA context and all handles/weights created in it.
// No tensor math or blend order changes; only independent spatial tiles move.
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};

unsafe extern "C" {
    fn cuCtxGetCurrent(context: *mut *mut c_void) -> i32;
    fn cuMemcpyPeerAsync(dst: u64, dst_context: *mut c_void, src: u64,
                         src_context: *mut c_void, bytes: usize, stream: Stream) -> i32;
    fn cuDeviceGetUuid(uuid: *mut [u8;16], device: i32) -> i32;
}

fn assigned_tiles(rank:usize,tiles:usize,active:usize)->std::iter::StepBy<std::ops::Range<usize>>{(rank..tiles).step_by(active)}
/// Workers only: when the coordinator computes no tiles, rank r>=1 owns r-1, r-1+(active-1), ...
fn worker_tiles(rank:usize,tiles:usize,active:usize,coordinator:bool)->std::iter::StepBy<std::ops::Range<usize>>{
    if coordinator||active<2{assigned_tiles(rank,tiles,active)}else{((rank-1)..tiles).step_by(active-1)}
}
/// The coordinator gives its tiles to the workers when that does not raise the
/// per-GPU maximum, leaving it free for stitching and streaming.
fn coordinator_computes(tiles:usize,active:usize)->bool{
    if std::env::var("H3_VAE_COORDINATOR_TILES").as_deref()==Ok("1"){return true;}
    active<2||tiles.div_ceil(active-1)>tiles.div_ceil(active)
}

fn current_context() -> Result<usize> {
    let mut context=ptr::null_mut();check(unsafe{cuCtxGetCurrent(&mut context)})?;
    if context.is_null(){return Err("missing VAE CUDA context".into());}Ok(context as usize)
}
fn device_uuid(rank: usize) -> Result<String> {
    let mut uuid=[0u8;16];check(unsafe{cuDeviceGetUuid(&mut uuid,rank as i32)})?;
    let s=uuid.iter().map(|v|format!("{v:02x}")).collect::<String>();
    Ok(format!("GPU-{}-{}-{}-{}-{}",&s[..8],&s[8..12],&s[12..16],&s[16..20],&s[20..]))
}

#[derive(Clone)]
struct TileJob {
    model_root: PathBuf, stage: &'static str, source: u64, source_context: usize,
    source_bytes: usize, target: u64, target_context: usize, tile_bytes: usize,
    width: usize, height: usize, latent_frames: usize, th: usize, tw: usize,
    ys: Vec<i32>, xs: Vec<i32>, world: usize, coordinator: bool,
    /// Queued decode: bytes of one chunk's tile buffer on GPU0 (there are QUEUE_SLOTS); 0 otherwise.
    slot_bytes: usize,
}
impl TileJob {
    fn count(&self)->usize{self.ys.len()*self.xs.len()}
    fn slot_offset(&self,chunk:usize,index:usize)->usize{(chunk%QUEUE_SLOTS)*self.slot_bytes+index*self.tile_bytes}
}

/// Decode tiles go through one queue that GPU0 and every worker take from, in (chunk, tile)
/// order and across chunk boundaries: a GPU that finishes early takes the next tile instead of
/// waiting for the slowest, and GPU0 decodes tiles whenever it would otherwise wait.
/// H3_VAE_DECODE_QUEUE=0 restores the fixed per-chunk split. Tile outputs land in one of
/// QUEUE_SLOTS chunk buffers on GPU0; a chunk is handed out only once GPU0 has stitched the
/// chunk that used its buffer before. Each tile is computed exactly as in the fixed split, only on
/// whichever GPU takes it, and chunks are stitched in the original order.
const QUEUE_SLOTS:usize=2;
fn decode_queue_enabled()->bool{std::env::var("H3_VAE_DECODE_QUEUE").as_deref()!=Ok("0")}
struct QueueState{next:usize,stitched:usize,done:Vec<usize>,failed:Option<String>,progress:usize}
struct DecodeQueue{tiles:usize,chunks:usize,state:std::sync::Mutex<QueueState>,changed:std::sync::Condvar}
enum Take{Tile(usize,usize),Wait,Exhausted}
impl DecodeQueue {
    fn new(tiles:usize,chunks:usize)->Self{
        Self{tiles,chunks,changed:std::sync::Condvar::new(),
            state:std::sync::Mutex::new(QueueState{next:0,stitched:0,done:vec![0;chunks],failed:None,progress:0})}
    }
    fn lock(&self)->Result<std::sync::MutexGuard<'_,QueueState>>{self.state.lock().map_err(|_|"VAE decode queue poisoned".into())}
    fn exhausted(&self,s:&QueueState)->bool{s.next>=self.tiles*self.chunks}
    /// Whether the next tile's chunk buffer is free (GPU0 has stitched the chunk that last used it).
    fn available(&self,s:&QueueState)->bool{!self.exhausted(s)&&s.next/self.tiles<s.stitched+QUEUE_SLOTS}
    fn take(&self)->Result<Take>{
        let mut s=self.lock()?;
        if s.failed.is_some()||self.exhausted(&s){return Ok(Take::Exhausted);}
        if !self.available(&s){return Ok(Take::Wait);}
        let item=s.next;s.next+=1;Ok(Take::Tile(item/self.tiles,item%self.tiles))
    }
    fn update(&self,change:impl FnOnce(&mut QueueState))->Result<()>{
        let mut s=self.lock()?;change(&mut s);s.progress+=1;drop(s);self.changed.notify_all();Ok(())
    }
    /// A tile's output has landed in GPU0's chunk buffer.
    fn complete(&self,chunk:usize)->Result<()>{self.update(|s|s.done[chunk]+=1)}
    /// GPU0 has consumed this chunk's buffer: the chunk QUEUE_SLOTS later may now be handed out.
    fn stitched(&self,chunk:usize)->Result<()>{self.update(|s|s.stitched=chunk+1)}
    fn fail(&self,error:&str){let _=self.update(|s|{if s.failed.is_none(){s.failed=Some(error.to_owned());}});}
    fn chunk_complete(&self,chunk:usize)->Result<bool>{
        let s=self.lock()?;if let Some(error)=&s.failed{return Err(error.clone().into());}Ok(s.done[chunk]==self.tiles)
    }
    /// Waits until `ready` holds or the queue fails. No progress at all for 120 s means a GPU
    /// stopped without reporting: fail the queue so every participant returns.
    fn wait_until(&self,ready:impl Fn(&Self,&QueueState)->bool)->Result<()>{
        let mut s=self.lock()?;let mut seen=s.progress;let mut since=Instant::now();
        while s.failed.is_none()&&!ready(self,&s){
            s=self.changed.wait_timeout(s,std::time::Duration::from_secs(1)).map_err(|_|"VAE decode queue poisoned")?.0;
            if s.progress!=seen{seen=s.progress;since=Instant::now();}
            else if since.elapsed().as_secs()>=120{s.failed=Some("VAE decode queue stalled".into());self.changed.notify_all();}
        }
        match &s.failed{Some(error)=>Err(error.clone().into()),None=>Ok(())}
    }
}
unsafe extern "C" {
    fn cuEventCreate(event:*mut *mut c_void,flags:u32)->i32;
    fn cuEventRecord(event:*mut c_void,stream:Stream)->i32;
    fn cuEventSynchronize(event:*mut c_void)->i32;
    fn cuEventDestroy_v2(event:*mut c_void)->i32;
}
/// Completion of one queued tile on a worker stream: its peer copy into GPU0 has landed.
struct TileEvent(*mut c_void);
impl TileEvent {
    // Blocking sync, no timing: a waiting worker sleeps instead of spinning a CPU core.
    fn new()->Result<Self>{let mut event=ptr::null_mut();check(unsafe{cuEventCreate(&mut event,0x1|0x2)})?;Ok(Self(event))}
    fn record(&self,stream:Stream)->Result<()>{check(unsafe{cuEventRecord(self.0,stream)})}
    fn wait(&self)->Result<()>{check(unsafe{cuEventSynchronize(self.0)})}
}
impl Drop for TileEvent{fn drop(&mut self){unsafe{cuEventDestroy_v2(self.0);}}}
enum TileCommand {Prepare(Arc<TileJob>), Run(usize), Queue(Arc<DecodeQueue>), Finish, ReleaseIdleWorkspace, Stop}
type TileReply=std::result::Result<Value,String>;
struct TileWorker {command:mpsc::Sender<TileCommand>,reply:mpsc::Receiver<TileReply>,thread:Option<JoinHandle<()>>}
struct TileTeam {workers:Vec<TileWorker>,world:usize,active:usize,identity:Vec<Value>,completed:usize,coordinator:bool,
    /// The last decode used the work queue; tiles decoded per GPU (GPU0 first).
    queued:bool,queued_tiles:Vec<usize>}

struct TileState {encoder:Option<Context>,decoder:Option<Context>,job:Option<Arc<TileJob>>,input:u64,prequant:u64,nonfinite:u64,rank:usize}
// Idle/partial-Prepare cleanup releases both loaded contexts. Only the stage
// whose active job was taken can complete, once; retries have no active job.
fn finish_tile_contexts<T>(encoder:Option<&mut T>,decoder:Option<&mut T>,active:Option<&str>,
    mut release:impl FnMut(&mut T)->Result<()>,mut completed:impl FnMut(&mut T))->Result<()>{
    let mut error=None;
    for (stage,context) in [("encoder",encoder),("decoder",decoder)]{
        if let Some(context)=context{match release(context){
            Ok(())=>if active==Some(stage){completed(context);},
            Err(value)=>if error.is_none(){error=Some(value);},
        }}
    }
    if let Some(error)=error{Err(error)}else{Ok(())}
}
impl TileState {
    fn release_idle_workspace(&mut self)->Result<Value>{
        if self.job.is_some(){return Err("cannot release active tile worker workspace".into());}
        let mut components=serde_json::Map::new();let mut released=0u64;
        for (name,ctx) in [("encoder",self.encoder.as_mut()),("decoder",self.decoder.as_mut())]{
            if let Some(ctx)=ctx{let report=ctx.release_idle_workspace()?;released+=report["released_bytes"].as_u64().unwrap();components.insert(name.into(),report);}
        }
        let report=release_dense_workspace()?;released+=report["released_bytes"].as_u64().unwrap();components.insert("thread_sdpa".into(),report);
        Ok(json!({"rank":self.rank,"released_bytes":released,"components":components}))
    }
    fn new(rank:usize)->Self{Self{encoder:None,decoder:None,job:None,input:0,prequant:0,nonfinite:0,rank}}
    fn context(&mut self,stage:&str)->Result<&mut Context>{
        (if stage=="decoder"{&mut self.decoder}else{&mut self.encoder}).as_mut().ok_or_else(||"VAE tile context missing".into())
    }
    fn prepare(&mut self,job:Arc<TileJob>)->Result<Value>{
        let mut mark=std::time::Instant::now();let mut split=serde_json::Map::new();
        let profiled=std::env::var_os("H3_VAE_DECODE_PROFILE_DIR").is_some();
        let mut lap=|name:&str|{if profiled{split.insert(name.into(),json!(mark.elapsed().as_secs_f64()));mark=std::time::Instant::now();}};
        self.finish()?;
        lap("release_previous");
        let rank=self.rank;
        let slot=if job.stage=="decoder"{&mut self.decoder}else{&mut self.encoder};
        if slot.is_none(){*slot=Some(Context::load(&job.model_root,job.stage,vec![job.width,job.height])?);}
        let ctx=slot.as_mut().unwrap();
        if ctx.model_root!=job.model_root{return Err("VAE tile worker model changed".into());}
        if profiled {
            // The same steps as select_spatial, timed separately.
            lap("context");
            ctx.program.arena.sync()?;lap("plans_sync");
            ctx.program.ensure_workspace()?;lap("plans_workspace");
            ctx.program.manifest=ctx.program.spatial.select(job.stage,job.th,job.tw)?;lap("plans_select");
        }else{ctx.program.select_spatial(job.stage,job.th,job.tw)?;}
        let input=ctx.program.arena.alloc(job.source_bytes)?;
        let prequant=if job.stage=="decoder"{ctx.program.arena.alloc(24*7*(job.th/16)*(job.tw/16)*4)?}else{0};
        let nonfinite=allocate_upload(&mut ctx.program.arena,&[0,0,0,0])?;
        lap("allocations");
        let context=current_context()?;
        check(unsafe{cuMemcpyPeerAsync(input,context as _,job.source,job.source_context as _,job.source_bytes,ctx.program.arena.stream)})?;
        ctx.program.arena.sync()?;
        lap("latent_peer_copy");
        let result=json!({"rank":rank,"gpu_uuid":device_uuid(rank)?,"thread_id":format!("{:?}",thread::current().id()),
            "context_identity":context,"stage":job.stage,"model_load_count":1,"requests_completed":ctx.requests_completed,"resident_model_load_seconds":ctx.load_seconds,
            "tile_count":worker_tiles(rank,job.count(),job.world,job.coordinator).count(),"spatial_plans":ctx.program.spatial_report(),
            "prepare_split_seconds":split});
        self.input=input;self.prequant=prequant;self.nonfinite=nonfinite;self.job=Some(job);Ok(result)
    }
    fn run(&mut self,chunk:usize)->Result<Value>{
        let job=self.job.clone().ok_or("VAE tile request not prepared")?;
        let (input,prequant,nonfinite,rank)=(self.input,self.prequant,self.nonfinite,self.rank);
        let ctx=self.context(job.stage)?;let program=&mut ctx.program;
        let tile_in=input_pointer(program)?;let tile_out=output_pointer(program)?;
        program.arena.upload(nonfinite,&[0,0,0,0])?;
        let context=current_context()?;let mut completed=0;
        for index in worker_tiles(rank,job.count(),job.world,job.coordinator){
            let (y,x)=(job.ys[index/job.xs.len()],job.xs[index%job.xs.len()]);
            if job.stage=="decoder"{
                check(unsafe{h3vg_extract(input as _,prequant as _,24,job.latent_frames as _,(job.height/16) as _,(job.width/16) as _,(chunk*5) as _,y/16,x/16,7,(job.th/16) as _,(job.tw/16) as _,program.arena.stream)})?;
                ctx.post.as_ref().ok_or("VAE tile postquant missing")?.forward_spatial(&program.arena,prequant,tile_in,job.th/16,job.tw/16)?;
            }else{
                check(unsafe{h3vg_extract(input as _,tile_in as _,3,1,job.height as _,job.width as _,0,y,x,1,job.th as _,job.tw as _,program.arena.stream)})?;
            }
            program.run(false)?;
            if job.stage=="decoder"{check(unsafe{h3v_half_nonfinite(tile_out as _,nonfinite as _,(job.tile_bytes/2) as _,program.arena.stream)})?;}
            check(unsafe{cuMemcpyPeerAsync(job.target+(index*job.tile_bytes) as u64,job.target_context as _,tile_out,context as _,job.tile_bytes,program.arena.stream)})?;
            completed+=1;
        }
        let bad=u32::from_le_bytes(program.arena.download(nonfinite,4)?.try_into().unwrap());
        if bad!=0{return Err(format!("nonfinite VAE worker tile values: {bad}").into());}
        Ok(json!({"rank":rank,"tiles":completed,"nonfinite":bad}))
    }
    /// Decodes queued tiles until the queue is exhausted; any error fails the whole queue.
    fn run_queue(&mut self,queue:&DecodeQueue)->Result<Value>{
        let result=self.queue_tiles(queue);
        if let Err(error)=&result{queue.fail(&error.to_string());}
        result
    }
    fn queue_tiles(&mut self,queue:&DecodeQueue)->Result<Value>{
        let job=self.job.clone().ok_or("VAE tile request not prepared")?;
        if job.stage!="decoder"||job.slot_bytes==0{return Err("only decode tiles are queued".into());}
        let (input,prequant,nonfinite,rank)=(self.input,self.prequant,self.nonfinite,self.rank);
        let ctx=self.context(job.stage)?;let program=&mut ctx.program;
        let tile_in=input_pointer(program)?;let tile_out=output_pointer(program)?;
        program.arena.upload(nonfinite,&[0,0,0,0])?;
        let context=current_context()?;let started=Instant::now();
        // At most two tiles in flight: the GPU always has the next one queued, and a tile is taken
        // only when this GPU is about to be free, so faster GPUs end up with more tiles.
        let events=[TileEvent::new()?,TileEvent::new()?];
        let mut inflight=std::collections::VecDeque::<(usize,usize)>::new();
        let settle=|inflight:&mut std::collections::VecDeque<(usize,usize)>|->Result<()>{
            let (chunk,event)=inflight.pop_front().ok_or("no tile in flight")?;events[event].wait()?;queue.complete(chunk)
        };
        let (mut computed,mut waited)=(0usize,0f64);
        loop{match queue.take()?{
            Take::Tile(chunk,index)=>{
                if inflight.len()==2{settle(&mut inflight)?;}
                let (y,x)=(job.ys[index/job.xs.len()],job.xs[index%job.xs.len()]);
                check(unsafe{h3vg_extract(input as _,prequant as _,24,job.latent_frames as _,(job.height/16) as _,(job.width/16) as _,(chunk*5) as _,y/16,x/16,7,(job.th/16) as _,(job.tw/16) as _,program.arena.stream)})?;
                ctx.post.as_ref().ok_or("VAE tile postquant missing")?.forward_spatial(&program.arena,prequant,tile_in,job.th/16,job.tw/16)?;
                program.run(false)?;
                check(unsafe{h3v_half_nonfinite(tile_out as _,nonfinite as _,(job.tile_bytes/2) as _,program.arena.stream)})?;
                check(unsafe{cuMemcpyPeerAsync(job.target+job.slot_offset(chunk,index) as u64,job.target_context as _,tile_out,context as _,job.tile_bytes,program.arena.stream)})?;
                let event=if inflight.front().is_some_and(|&(_,e)|e==0){1}else{0};
                events[event].record(program.arena.stream)?;inflight.push_back((chunk,event));computed+=1;
            },
            // Never block with tiles in flight: GPU0 may be waiting for exactly those.
            Take::Wait=>if inflight.is_empty(){
                let waiting=Instant::now();queue.wait_until(|q,s|q.available(s)||q.exhausted(s))?;waited+=waiting.elapsed().as_secs_f64();
            }else{settle(&mut inflight)?;},
            Take::Exhausted=>break,
        }}
        while !inflight.is_empty(){settle(&mut inflight)?;}
        let bad=u32::from_le_bytes(program.arena.download(nonfinite,4)?.try_into().unwrap());
        if bad!=0{return Err(format!("nonfinite VAE worker tile values: {bad}").into());}
        Ok(json!({"rank":rank,"tiles":computed,"nonfinite":bad,"wall_seconds":started.elapsed().as_secs_f64(),"wait_seconds":waited}))
    }
    fn finish(&mut self)->Result<Value>{
        let active=self.job.take().map(|job|job.stage);
        finish_tile_contexts(self.encoder.as_mut(),self.decoder.as_mut(),active,
            |ctx|{ctx.program.arena.release_after(ctx.persistent_mark)?;ctx.program.workspace_released();Ok(())},|ctx|ctx.requests_completed+=1)?;
        Ok(json!({"finished":true,"rank":self.rank}))
    }
}
impl TileTeam {
    fn configured_world()->Result<usize>{
        Self::parse_world(&std::env::var("H3_VIDEO_VAE_WORLD_SIZE").unwrap_or_else(|_|"1".into()))
    }
    fn parse_world(value:&str)->Result<usize>{
        let n=value.parse::<usize>()?;
        if ![1,2,4,8].contains(&n){return Err("VAE world size must be 1, 2, 4, or 8".into());}Ok(n)
    }
    fn new(world:usize)->Result<Self>{
        let mut team=Self{workers:vec![],world,active:0,identity:vec![],completed:0,coordinator:true,queued:false,queued_tiles:vec![]};
        for rank in 1..world{
            let (tx,rx)=mpsc::channel();let (reply_tx,reply_rx)=mpsc::channel();
            let handle=thread::Builder::new().name(format!("h3-vae-{rank}")).spawn(move||{
                h3_router::select_device(rank as i32);
                let mut state=TileState::new(rank);
                while let Ok(command)=rx.recv(){
                    let result=match command{TileCommand::Prepare(job)=>state.prepare(job),TileCommand::Run(chunk)=>state.run(chunk),
                        TileCommand::Queue(queue)=>state.run_queue(&queue),TileCommand::Finish=>state.finish(),
                        TileCommand::ReleaseIdleWorkspace=>state.release_idle_workspace(),TileCommand::Stop=>break};
                    if reply_tx.send(result.map_err(|e|e.to_string())).is_err(){break;}
                }
                let _=state.finish();
            })?;
            team.workers.push(TileWorker{command:tx,reply:reply_rx,thread:Some(handle)});
        }
        Ok(team)
    }
    fn prepare(&mut self,mut job:TileJob)->Result<()> {
        self.active=self.world.min(job.count());job.world=self.active;
        // The encoder stage keeps its original assignment; only decode hands off.
        job.coordinator=job.stage!="decoder"||coordinator_computes(job.count(),self.active);
        self.coordinator=job.coordinator;
        let coordinator=json!({"rank":0,"gpu_uuid":device_uuid(0)?,"thread_id":format!("{:?}",thread::current().id()),
            "context_identity":current_context()?,"stage":job.stage,"model_load_count":1,"tile_count":if job.coordinator{assigned_tiles(0,job.count(),job.world).count()}else{0}});
        let job=Arc::new(job);let mut sent=0;
        for worker in self.workers.iter().take(self.active-1){
            if worker.command.send(TileCommand::Prepare(job.clone())).is_err(){
                self.drain(sent)?;return Err("VAE tile worker disconnected during prepare".into());
            }sent+=1;
        }
        let mut identity=vec![coordinator];
        let remote=self.drain(sent)?;identity.extend(remote);self.identity=identity;Ok(())
    }
    fn drain(&self,count:usize)->Result<Vec<Value>>{
        let mut values=Vec::new();let mut error=None;
        for worker in self.workers.iter().take(count){match worker.reply.recv(){
            Ok(Ok(v))=>values.push(v),Ok(Err(e))=>{if error.is_none(){error=Some(e)}},Err(e)=>{if error.is_none(){error=Some(e.to_string())}}
        }}
        if let Some(e)=error{Err(e.into())}else{Ok(values)}
    }
    /// Every active worker decodes queued tiles until the queue is exhausted; each replies once.
    fn start_queue(&self,queue:&Arc<DecodeQueue>)->Result<usize>{
        let mut sent=0;
        for worker in self.workers.iter().take(self.active-1){
            if worker.command.send(TileCommand::Queue(queue.clone())).is_err(){
                queue.fail("VAE tile worker disconnected during run");self.drain(sent)?;return Err("VAE tile worker disconnected during run".into());
            }
            sent+=1;
        }Ok(sent)
    }
    fn dispatch(&self,chunk:usize)->Result<usize>{
        let mut sent=0;
        for worker in self.workers.iter().take(self.active-1){
            if worker.command.send(TileCommand::Run(chunk)).is_err(){self.drain(sent)?;return Err("VAE tile worker disconnected during run".into());}
            sent+=1;
        }Ok(sent)
    }
    fn finish(&mut self)->Result<()> {
        let mut sent=vec![];let mut error=None;
        for (index,worker) in self.workers.iter().take(self.active.saturating_sub(1)).enumerate(){
            if worker.command.send(TileCommand::Finish).is_ok(){sent.push(index);}else{error=Some("VAE worker disconnected during cleanup".to_string());}
        }
        for index in sent{match self.workers[index].reply.recv(){
            Ok(Ok(_))=>{},Ok(Err(e))=>{if error.is_none(){error=Some(e);}},Err(_)=>{if error.is_none(){error=Some("VAE worker cleanup reply missing".into());}}
        }}
        self.active=0;
        if let Some(e)=error{Err(e.into())}else{self.completed+=1;Ok(())}
    }
    fn active_count(&self)->usize{self.active}
    fn release_idle_workspace(&mut self)->Result<Value>{
        if self.active!=0{return Err("cannot release active video tile team workspace".into());}
        let mut sent=0;
        // Include every loaded worker, even if the last small shape used fewer ranks.
        for worker in &self.workers{
            if worker.command.send(TileCommand::ReleaseIdleWorkspace).is_err(){self.drain(sent)?;return Err("VAE tile worker disconnected during workspace release".into());}sent+=1;
        }
        let workers=self.drain(sent)?;let released:u64=workers.iter().map(|r|r["released_bytes"].as_u64().unwrap()).sum();
        Ok(json!({"released_bytes":released,"workers":workers}))
    }
    fn coordinator_tiles(&self)->bool{self.coordinator}
    fn report(&self)->Value{json!({"world_size":self.world,"active_workers":self.identity.len(),
        "workers":self.identity,"requests_completed":self.completed,"transport":"CUDA peer copy",
        "partition":if self.queued{"work queue in (chunk, tile) order across chunks; the coordinator decodes tiles while it waits"}
            else if self.coordinator{"row-major tile index modulo active workers"}else{"row-major tile index modulo workers; coordinator stitches only"},
        "queued_tiles_per_gpu":self.queued_tiles,
        "coordinator_computes_tiles":self.coordinator,"next_chunk_dispatched_before_emit":true,"original_stitch_order":true,"resident_weights":true})}
}
impl Drop for TileTeam {
    fn drop(&mut self){
        for worker in &self.workers{let _=worker.command.send(TileCommand::Stop);}
        for worker in &mut self.workers{if let Some(handle)=worker.thread.take(){let _=handle.join();}}
    }
}
// Even on a coordinator kernel/file error, remote writes must finish before
// Resident releases GPU0's request allocations. The outstanding replies have
// already been drained by the chunk dispatcher before any error propagates.
struct TileSession<'a>{team:&'a mut TileTeam,finished:bool}
impl TileSession<'_>{fn finish(&mut self)->Result<()>{let result=self.team.finish();self.finished=true;result}}
impl Drop for TileSession<'_>{fn drop(&mut self){if !self.finished{let _=self.team.finish();}}}

#[cfg(test)]
mod completion_counter_tests {
    use super::*;
    #[derive(Default)]struct State{completed:usize,released:usize,fail:bool}
    fn finish(encoder:&mut State,decoder:&mut State,active:&mut Option<&str>)->Result<()>{
        finish_tile_contexts(Some(encoder),Some(decoder),active.take(),|state|{
            state.released+=1;if state.fail{Err("injected release error".into())}else{Ok(())}
        },|state|state.completed+=1)
    }
    #[test]
    fn idle_partial_prepare_and_repeated_finish_do_not_complete_jobs(){
        let (mut encoder,mut decoder)=(State::default(),State::default());
        finish(&mut encoder,&mut decoder,&mut None).unwrap();
        let mut active=Some("encoder");finish(&mut encoder,&mut decoder,&mut active).unwrap();
        finish(&mut encoder,&mut decoder,&mut active).unwrap();
        assert_eq!((encoder.completed,decoder.completed),(1,0));
        let mut active=Some("decoder");finish(&mut encoder,&mut decoder,&mut active).unwrap();
        assert_eq!((encoder.completed,decoder.completed),(1,1));
        assert_eq!((encoder.released,decoder.released),(4,4));
    }
    #[test]
    fn release_failure_still_cleans_other_context_without_false_completion(){
        let (mut encoder,mut decoder)=(State{fail:true,..State::default()},State::default());
        let mut active=Some("encoder");assert!(finish(&mut encoder,&mut decoder,&mut active).is_err());
        assert_eq!((encoder.completed,decoder.completed),(0,0));
        assert_eq!((encoder.released,decoder.released),(1,1));
        encoder.fail=false;finish(&mut encoder,&mut decoder,&mut active).unwrap();
        assert_eq!((encoder.completed,decoder.completed),(0,0));
    }
}
