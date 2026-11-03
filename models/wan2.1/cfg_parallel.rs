//! GPU 1 owns the unconditional DiT pass. Only exact peer copies cross GPUs;
//! each pass keeps the single-GPU operator order and the scheduler stays on GPU 0.
//! At cold start GPU 1 copies GPU 0's DiT weights over the peer link instead of
//! reading them again.
use crate::{cuda::{self,Buffer,Device,Event},dit::{Dit,Grid,PassContext},serving::Session,
    weights::DeviceWeights,Result};
use native_core_infra::{backend_service::Session as ResidentSession,model_load::Loaded};
use serde_json::{json,Value};
use std::{cell::OnceCell,path::PathBuf,rc::Rc,sync::mpsc,thread,time::Instant};

type Reply=mpsc::Sender<std::result::Result<Value,String>>;
enum Command{
    Warm(PathBuf,Reply),Prefetch(PathBuf,usize,Option<Vec<String>>,Reply),
    Prewarm(Grid,Reply),Replicate(usize,Grid,Reply),
    Configure(Grid,u64,usize,usize,u64,usize,u64,usize,Reply),
    Forward(u64,i64,u64,Event,Reply),Retire(Reply),Evict(Reply),Audit(Reply),Status(Reply),Stop,
}
pub(crate) struct Pending{reply:Option<mpsc::Receiver<std::result::Result<Value,String>>>}
impl Pending{pub(crate) fn finish(mut self)->Result<Value>{self.reply.take().ok_or("Wan peer reply consumed")?
    .recv().map_err(|_|"Wan GPU 1 exited")?.map_err(Into::into)}}
impl Drop for Pending{fn drop(&mut self){if let Some(reply)=self.reply.take(){let _=reply.recv();}}}
/// GPU 1's copy of GPU 0's DiT in flight. Dropping it waits for the copy, which reads `_source`.
pub(crate) struct Replica{pending:Pending,_source:Rc<Loaded>}
impl Replica{pub(crate) fn finish(self)->Result<Value>{self.pending.finish()}}
struct State{
    // Drop views/buffers before their weight owner and CUDA context.
    context:Option<PassContext>,y:Option<Buffer>,latent:Option<Buffer>,out:Option<Buffer>,
    capture:Option<native_core_infra::execution_start::Capture>,first_dit:Option<Value>,
    session:Session,forwards:usize,
}
impl State{
    fn copy(&self,address:u64,bytes:usize)->Result<Buffer>{
        let value=Buffer::new(bytes)?;
        cuda::check(unsafe{cuda::memcpy_peer(value.dev(),1,address,0,bytes,self.session.device.stream)},"Wan CFG input peer copy")?;
        Ok(value)
    }
    #[allow(clippy::too_many_arguments)]
    fn configure(&mut self,grid:Grid,context:u64,context_bytes:usize,rows:usize,clip:u64,clip_bytes:usize,y:u64,y_bytes:usize)->Result<Value>{
        self.session.device.sync()?;
        self.context.take();self.y.take();self.latent.take();self.out.take();
        self.capture.take();self.first_dit=None;
        let device:&'static Device=unsafe{&*(&*self.session.device as *const Device)};
        if self.session.dit.is_none(){
            let loaded=self.session.gpu.get("dit").ok_or("Wan CFG DiT weights not prepared")?.clone();
            self.session.dit=Some(Dit::load_from(&DeviceWeights{loaded,prefix:"dit.".into()},grid,device)?);
        }
        self.session.dit.as_mut().unwrap().set_grid(grid)?;
        let(context,clip)=(self.copy(context,context_bytes)?,self.copy(clip,clip_bytes)?);
        self.context=Some(self.session.dit.as_ref().unwrap().pass_context(context.dev(),rows,clip.dev())?);
        self.y=Some(self.copy(y,y_bytes)?);
        self.latent=Some(Buffer::new(grid.latent_elements(16)*4)?);
        self.out=Some(Buffer::new(grid.latent_elements(16)*4)?);
        self.session.device.sync()?;self.capture=Some(native_core_infra::execution_start::Capture::begin());
        Ok(json!({"configured":true}))
    }
    fn replicate(&mut self,source:&Loaded,grid:Grid)->Result<Value>{
        let started=Instant::now();
        let loaded=Rc::new(unsafe{source.replicate(0,self.session.device.stream)}?);
        let copied=started.elapsed().as_secs_f64();
        let device:&'static Device=unsafe{&*(&*self.session.device as *const Device)};
        self.session.dit=None;self.session.gpu.insert("dit".into(),loaded.clone());
        self.session.dit=Some(Dit::load_from(&DeviceWeights{loaded:loaded.clone(),prefix:"dit.".into()},grid,device)?);
        self.session.device.sync()?;
        Ok(json!({"device_uuid":self.session.uuid()?,"bytes":loaded.stats.device_bytes,"copy_seconds":copied,
            "ready_seconds":started.elapsed().as_secs_f64()}))
    }
    fn forward(&mut self,latent:u64,t:i64,out:u64,ready:Event)->Result<Value>{
        let device=&self.session.device;let input=self.latent.as_ref().ok_or("Wan CFG input missing")?;
        ready.wait_on(device.stream)?;
        cuda::check(unsafe{cuda::memcpy_peer(input.dev(),1,latent,0,input.bytes(),device.stream)},"Wan CFG latent peer copy")?;
        let dit=self.session.dit.as_ref().ok_or("Wan CFG DiT missing")?;
        dit.time(t)?;
        dit.forward(input.dev(),self.y.as_ref().ok_or("Wan CFG y missing")?.dev(),
            self.context.as_ref().ok_or("Wan CFG context missing")?,self.out.as_ref().unwrap().dev(),&mut|_,_|Ok(()))?;
        cuda::check(unsafe{cuda::memcpy_peer(out,0,self.out.as_ref().unwrap().dev(),1,input.bytes(),device.stream)},"Wan CFG prediction peer copy")?;
        device.sync()?;self.forwards+=1;
        if self.first_dit.is_none(){self.first_dit=self.capture.as_ref().ok_or("Wan peer capture missing")?.report()?;}
        Ok(json!({"device_uuid":self.session.uuid()?,"unconditional_forwards":self.forwards}))
    }
    fn release_request(&mut self)->Result<()>{
        self.session.device.sync()?;self.context.take();self.y.take();self.latent.take();self.out.take();self.capture.take();Ok(())
    }
    fn status(&self)->Result<Value>{let mut value=self.session.metadata()?;value["unconditional_forwards"]=json!(self.forwards);
        value["first_dit_start"]=json!(self.first_dit);Ok(value)}
}
pub(crate) struct CfgWorker{sender:mpsc::Sender<Command>,thread:Option<thread::JoinHandle<()>>,ready:OnceCell<Event>}
impl CfgWorker{
    pub(crate) fn open()->Result<Self>{
        cuda::enable_peer_access(1)?;let ready=Event::new()?;
        let(worker,started)=Self::spawn()?;
        started.recv().map_err(|_|"Wan GPU 1 startup closed")?.map_err(|e:String|e)?;
        worker.ready.set(ready).map_err(|_|"Wan GPU 1 event set twice")?;Ok(worker)
    }
    /// Opens GPU 1 on its own thread and returns at once, so its CUDA context comes up while
    /// GPU 0's does. A GPU 1 failure surfaces on the first command waited for.
    pub(crate) fn start()->Result<Self>{Ok(Self::spawn()?.0)}
    fn spawn()->Result<(Self,mpsc::Receiver<std::result::Result<(),String>>)>{
        let(sender,receiver)=mpsc::channel();let(boot,started)=mpsc::channel();
        let thread=thread::Builder::new().name("wan21-cfg-gpu1".into()).spawn(move||{
            let run=(||->Result<()>{
                let session=Session::open_rank(1)?;cuda::enable_peer_access(0)?;
                let mut state=State{context:None,y:None,latent:None,out:None,capture:None,first_dit:None,session,forwards:0};
                let _=boot.send(Ok(()));
                macro_rules! reply{($sender:expr,$result:expr)=>{{
                    let result=$result.map_err(|e|e.to_string());
                    let failed=result.is_err();
                    // The caller owns every GPU 0 pointer borrowed by this
                    // command. Drain queued peer copies before releasing that
                    // lease even when a later GPU operation rejects the call.
                    if failed{let _=state.session.device.sync();}
                    let _=$sender.send(result);
                    if failed{break;}
                }}}
                while let Ok(command)=receiver.recv(){match command{
                    Command::Warm(path,to)=>reply!(to,state.session.warm(&path)),
                    Command::Prefetch(path,budget,names,to)=>reply!(to,state.session.prefetch_components(&path,budget,names.as_deref())),
                    Command::Prewarm(grid,to)=>reply!(to,Dit::prewarm(grid).map(|_|json!({"prewarmed":true}))),
                    // The sender keeps the source allocation alive until this reply.
                    Command::Replicate(source,grid,to)=>reply!(to,state.replicate(unsafe{&*(source as *const Loaded)},grid)),
                    Command::Configure(grid,context,bytes,rows,clip,clip_bytes,y,y_bytes,to)=>
                        reply!(to,state.configure(grid,context,bytes,rows,clip,clip_bytes,y,y_bytes)),
                    Command::Forward(latent,t,out,ready,to)=>reply!(to,state.forward(latent,t,out,ready)),
                    Command::Retire(to)=>reply!(to,state.release_request().and_then(|_|state.session.retire_predictors())),
                    Command::Evict(to)=>reply!(to,state.release_request().and_then(|_|state.session.evict())),
                    Command::Audit(to)=>reply!(to,state.session.audit()),Command::Status(to)=>reply!(to,state.status()),
                    Command::Stop=>break,
                }}Ok(())
            })();
            if let Err(error)=run{eprintln!("Wan GPU 1: {error}");let _=boot.send(Err(error.to_string()));}
        })?;
        Ok((Self{sender,thread:Some(thread),ready:OnceCell::new()},started))
    }
    fn begin(&self,command:impl FnOnce(Reply)->Command)->Result<Pending>{
        let(to,reply)=mpsc::channel();self.sender.send(command(to)).map_err(|_|"Wan GPU 1 exited")?;
        Ok(Pending{reply:Some(reply)})
    }
    pub(crate) fn begin_warm(&self,path:&std::path::Path)->Result<Pending>{self.begin(|to|Command::Warm(path.into(),to))}
    pub(crate) fn begin_prefetch(&self,path:&std::path::Path,budget:usize,names:Option<&[String]>)->Result<Pending>{
        // A rank with only DiT may legitimately have no selected work yet.
        self.begin(|to|Command::Prefetch(path.into(),budget,names.map(|n|n.iter().filter(|n|n.as_str()=="dit").cloned().collect()),to))
    }
    /// Builds the request grid's attention plans on GPU 1 while GPU 0 still loads.
    pub(crate) fn begin_prewarm(&self,grid:Grid)->Result<Pending>{self.begin(|to|Command::Prewarm(grid,to))}
    /// GPU 1 copies `dit`, GPU 0's DiT weights, and builds its DiT for `grid`.
    pub(crate) fn begin_replicate(&self,dit:&Rc<Loaded>,grid:Grid)->Result<Replica>{
        Ok(Replica{pending:self.begin(|to|Command::Replicate(Rc::as_ptr(dit) as usize,grid,to))?,_source:dit.clone()})
    }
    pub(crate) fn configure(&self,grid:Grid,context:&Buffer,rows:usize,clip:&Buffer,y:&Buffer,device:&Device)->Result<()>{
        // Recorded on GPU 0's stream, so it is created on GPU 0's thread.
        if self.ready.get().is_none(){self.ready.set(Event::new()?).map_err(|_|"Wan GPU 1 event set twice")?;}
        device.sync()?;
        self.begin(|to|Command::Configure(grid,context.dev(),context.bytes(),rows,clip.dev(),clip.bytes(),y.dev(),y.bytes(),to))?.finish()?;Ok(())
    }
    pub(crate) fn forward(&self,latent:u64,t:i64,out:u64,device:&Device)->Result<Pending>{
        let ready=*self.ready.get().ok_or("Wan CFG peer not configured")?;
        ready.record(device.stream)?;self.begin(|to|Command::Forward(latent,t,out,ready,to))
    }
    pub(crate) fn begin_retire(&self)->Result<Pending>{self.begin(Command::Retire)}
    pub(crate) fn evict(&self)->Result<Value>{self.begin(Command::Evict)?.finish()}
    pub(crate) fn audit(&self)->Result<Value>{self.begin(Command::Audit)?.finish()}
    pub(crate) fn status(&self)->Result<Value>{self.begin(Command::Status)?.finish()}
}
impl Drop for CfgWorker{fn drop(&mut self){let _=self.sender.send(Command::Stop);if let Some(thread)=self.thread.take(){let _=thread.join();}}}
