// Per-request transport configuration, never process-global mutation.
use std::os::unix::net::UnixStream;
use std::net::Shutdown;
unsafe extern "C" {
    fn cuMemHostAlloc(pointer:*mut *mut c_void,bytes:usize,flags:u32)->i32;
    fn cuMemFreeHost(pointer:*mut c_void)->i32;
    fn cuMemAlloc_v2(pointer:*mut u64,bytes:usize)->i32;
    fn cuMemFree_v2(pointer:u64)->i32;
    fn h3vg_frame_sums(input:*const u8,output:*mut u64,count:i32,bytes:i64,stream:Stream)->i32;
    fn h3vg_delivery_resize(input:*const u8,scratch:*mut f32,output:*mut u8,count:i32,height:i32,width:i32,
        output_height:i32,output_width:i32,stream:Stream)->i32;
    fn cuMemcpyDtoHAsync_v2(dst:*mut c_void,src:u64,bytes:usize,stream:Stream)->i32;
    fn h3vg_pt8_range(input:*const c_void,output:*mut u8,h:i32,w:i32,frames:i32,first:i32,count:i32,stream:Stream)->i32;
}
#[derive(Clone)]
struct VideoRequest {
    stream_path:PathBuf,capture_raw:bool,
    delivery:Option<[usize;3]>,
    output_frames:Option<usize>,
    delivery_chunk_frames:usize,
}
impl VideoRequest {
    fn public_rgb_stream(path:&str)->Result<Self>{
        let stream_path=PathBuf::from(path);
        if !stream_path.is_absolute(){return Err("RGB stream path must be absolute".into());}
        Ok(Self{stream_path,capture_raw:false,delivery:None,output_frames:None,delivery_chunk_frames:22})
    }
    fn chunk_frames(&self)->usize {
        if self.delivery.is_some(){self.delivery_chunk_frames}else{22}
    }
    fn parse(value:&Value)->Result<Self>{
        if value["schema"]!="h3-native-request-context-v1"{return Err("unknown native request context schema".into());}
        let stream_path=PathBuf::from(value["video_stream_path"].as_str().ok_or("video stream path missing")?);
        if !stream_path.is_absolute(){return Err("video stream path must be absolute".into());}
        let delivery=if value["delivery"].is_null(){None}else{
            let d=&value["delivery"];
            let w=usize::try_from(d["width"].as_u64().ok_or("delivery width")?)?;
            let h=usize::try_from(d["height"].as_u64().ok_or("delivery height")?)?;
            let f=usize::try_from(d["num_frames"].as_u64().ok_or("delivery frames")?)?;
            if w==0||h==0||w%2!=0||h%2!=0||w>8192||h>8192||w*h>16_777_216||f==0{
                return Err("invalid delivery geometry".into());
            }
            Some([w,h,f])
        };
        let delivery_chunk_frames=value["delivery_chunk_frames"].as_u64().and_then(|n|usize::try_from(n).ok()).unwrap_or(22);
        if !(1..=128).contains(&delivery_chunk_frames){return Err("invalid delivery chunk capacity".into());}
        let output_frames=if value["short_video_delivery"].is_null(){None}else{
            let n=usize::try_from(value["short_video_delivery"]["num_frames"].as_u64().ok_or("invalid short delivery frames")?)?;
            if n==0||n>120||delivery.is_some(){return Err("invalid short video delivery".into());}Some(n)
        };
        Ok(Self{stream_path,capture_raw:value["capture_video_raw"].as_bool().unwrap_or(false),delivery,output_frames,delivery_chunk_frames})
    }
}
fn request_arguments(args:&[String])->Result<(Vec<String>,Option<VideoRequest>)>{
    let public=args.iter().enumerate().filter(|(_,v)|v.as_str()=="--rgb-stream").map(|(i,_)|i).collect::<Vec<_>>();
    if !public.is_empty(){
        if public.len()!=1||public[0]+2!=args.len()||args.iter().any(|v|v=="--request-context"){
            return Err("--rgb-stream must be one final PATH pair".into());
        }
        if args.get(1).is_none_or(|v|v!="decode"){
            return Err("--rgb-stream applies to ordinary decode only".into());
        }
        return Ok((args[..public[0]].to_vec(),Some(VideoRequest::public_rgb_stream(&args[public[0]+1])?)));
    }
    let mut clean=args.to_vec();let positions=clean.iter().enumerate().filter(|(_,v)|v.as_str()=="--request-context").map(|(i,_)|i).collect::<Vec<_>>();
    if positions.is_empty(){return Ok((clean,None));}
    if positions.len()!=1||positions[0]+2!=clean.len(){return Err("--request-context must be one final PATH pair".into());}
    let index=positions[0];let bytes=crate::startup_trace::read(&clean[index+1])?;
    if bytes.len()>65536{return Err("request context exceeds 64 KiB".into());}
    let request=VideoRequest::parse(&serde_json::from_slice(&bytes)?)?;clean.truncate(index);
    if clean.get(1).is_none_or(|v|v!="decode"&&v!="decode-long"&&v!="deliver-rgb"){return Err("video request context applies to decode/delivery only".into());}
    Ok((clean,Some(request)))
}
struct HostChunk {pointer:usize,capacity:usize}
impl HostChunk {
    fn new(capacity:usize)->Result<Self>{let mut p=ptr::null_mut();check(unsafe{cuMemHostAlloc(&mut p,capacity,0)})?;Ok(Self{pointer:p as usize,capacity})}
    fn download(&self,device:u64,bytes:usize,arena:&Arena)->Result<()>{
        if bytes>self.capacity{return Err("VAE pinned chunk capacity exceeded".into());}
        check(unsafe{cuMemcpyDtoHAsync_v2(self.pointer as _,device,bytes,arena.stream)})?;arena.sync()
    }
    fn bytes(&self,count:usize)->&[u8]{assert!(count<=self.capacity);unsafe{std::slice::from_raw_parts(self.pointer as *const u8,count)}}
}
impl Drop for HostChunk {fn drop(&mut self){unsafe{cuMemFreeHost(self.pointer as _);}}}
/// Page-locked stream chunks kept by the resident decoder across requests: pinning
/// ~70 MB chunks on every request cost ~0.13 s. Contents are always fully overwritten
/// before use; only the allocation is reused.
static HOST_CHUNK_POOL:std::sync::Mutex<Vec<HostChunk>>=std::sync::Mutex::new(Vec::new());
const HOST_CHUNK_POOL_LIMIT:usize=6;
impl HostChunk {
    fn pooled(capacity:usize)->Result<Self>{
        let reused={let mut pool=HOST_CHUNK_POOL.lock().map_err(|_|"pinned chunk pool poisoned")?;
            pool.iter().enumerate().filter(|(_,c)|c.capacity>=capacity).min_by_key(|(_,c)|c.capacity).map(|(i,_)|i).map(|i|pool.swap_remove(i))};
        match reused{Some(chunk)=>Ok(chunk),None=>HostChunk::new(capacity)}
    }
    fn recycle(chunks:Vec<HostChunk>){
        if let Ok(mut pool)=HOST_CHUNK_POOL.lock(){
            pool.extend(chunks);
            // Keep the largest chunks; dropping the rest frees them.
            pool.sort_by_key(|c|std::cmp::Reverse(c.capacity));pool.truncate(HOST_CHUNK_POOL_LIMIT);
        }
    }
}
enum FramePacket {Chunk{slot:usize,first:usize,count:usize,bytes:usize},End}
struct FrameHealth {device:u64,host:HostChunk,means:Vec<f64>}
impl FrameHealth {
    fn new(capacity:usize)->Result<Self>{
        let host=HostChunk::new(capacity*8)?;let mut device=0;
        check(unsafe{cuMemAlloc_v2(&mut device,capacity*8)})?;
        Ok(Self{device,host,means:Vec::new()})
    }
    fn enqueue(&self,input:u64,count:usize,pixels:usize,stream:Stream)->Result<()>{
        check(unsafe{h3vg_frame_sums(input as _,self.device as _,count as _,(pixels*3)as _,stream)})?;
        check(unsafe{cuMemcpyDtoHAsync_v2(self.host.pointer as _,self.device,count*8,stream)})
    }
    fn collect(&mut self,count:usize,pixels:usize){
        // emit's existing RGB download synchronized this stream already.
        for bytes in self.host.bytes(count*8).chunks_exact(8){
            self.means.push(u64::from_le_bytes(bytes.try_into().unwrap())as f64/(pixels*3)as f64);
        }
    }
}
impl Drop for FrameHealth{fn drop(&mut self){unsafe{cuMemFree_v2(self.device);}}}
struct FrameStream {
    sender:Option<mpsc::SyncSender<FramePacket>>,returned:mpsc::Receiver<usize>,writer:Option<JoinHandle<std::result::Result<(),String>>>,
    buffers:Vec<HostChunk>,available:Vec<usize>,frames:usize,next:usize,pixels:usize,capture:Option<fs::File>,raw_hash:Option<Sha256>,
    health:Option<FrameHealth>,health_error:Option<String>,
    request:VideoRequest,chunks:usize,
    /// Opt-in (H3_VAE_DECODE_PROFILE_DIR) synchronizing split of GPU0's per-chunk emit work.
    emit_profile:Option<[f64;3]>,
    resize_scratch:u64,resize_output:u64,capacity_frames:usize,
}
impl FrameStream {
    fn new(request:&VideoRequest,width:usize,height:usize,frames:usize,out:&Path)->Result<Self>{
        let frames=if let Some(n)=request.output_frames{if n>frames{return Err("delivery cannot exceed rendered frames".into());}n}else{frames};
        let [width,height,frames]=if let Some([w,h,f])=request.delivery{
            if f>frames{return Err("delivery cannot exceed rendered frames".into());}[w,h,f]
        }else{[width,height,frames]};
        let mut socket=UnixStream::connect(&request.stream_path)?;
        socket.set_write_timeout(Some(Duration::from_secs(120)))?;
        let mut header=b"H3RGB001".to_vec();for n in [width as u32,height as u32,frames as u32,0]{header.extend_from_slice(&n.to_le_bytes());}socket.write_all(&header)?;
        let capacity_frames=request.chunk_frames();
        let capacity=capacity_frames*width*height*3;let mut buffers=Vec::new();for _ in 0..3{buffers.push(HostChunk::pooled(capacity)?);}
        let pointers=buffers.iter().map(|b|b.pointer).collect::<Vec<_>>();
        let (sender,receiver)=mpsc::sync_channel::<FramePacket>(2);let (returned_tx,returned)=mpsc::channel();
        let writer=thread::Builder::new().name("h3-vae-stream".into()).spawn(move||->std::result::Result<(),String>{
            let mut next=0;
            while let Ok(packet)=receiver.recv(){match packet{
                FramePacket::Chunk{slot,first,count,bytes}=>{
                    if first!=next||count==0||first+count>frames{return Err("invalid VAE frame stream ordering".into());}
                    let mut prefix=Vec::with_capacity(16);prefix.extend_from_slice(&(first as u32).to_le_bytes());prefix.extend_from_slice(&(count as u32).to_le_bytes());prefix.extend_from_slice(&(bytes as u64).to_le_bytes());
                    socket.write_all(&prefix).and_then(|_|socket.write_all(unsafe{std::slice::from_raw_parts(pointers[slot] as *const u8,bytes)})).map_err(|_|"VAE stream consumer disconnected".to_string())?;
                    next+=count;let _=returned_tx.send(slot);
                },FramePacket::End=>{
                    if next!=frames{return Err("VAE stream frame count incomplete".into());}
                    let mut end=Vec::with_capacity(16);end.extend_from_slice(&(frames as u32).to_le_bytes());end.extend_from_slice(&0u32.to_le_bytes());end.extend_from_slice(&0u64.to_le_bytes());
                    socket.write_all(&end).and_then(|_|socket.shutdown(Shutdown::Write)).map_err(|_|"VAE stream completion failed".to_string())?;return Ok(());
                }
            }}Err("VAE producer aborted before stream completion".into())
        })?;
        let capture=if request.capture_raw{Some(fs::File::create(out.join("frames.raw"))?)}else{None};
        let (health,health_error)=match FrameHealth::new(capacity_frames){Ok(v)=>(Some(v),None),Err(e)=>(None,Some(e.to_string()))};
        Ok(Self{sender:Some(sender),returned,writer:Some(writer),buffers,available:vec![0,1,2],frames,next:0,pixels:width*height,capture,
            health,health_error,
            request:request.clone(),chunks:0,
            emit_profile:std::env::var_os("H3_VAE_DECODE_PROFILE_DIR").map(|_|[0.0;3]),
            resize_scratch:0,resize_output:0,capacity_frames,
            raw_hash:if request.capture_raw&&crate::debug_audit::enabled(){Some(Sha256::new())}else{None}})
    }
    fn emit(&mut self,mut device:u64,first:usize,mut count:usize,arena:&Arena,height:usize,width:usize)->Result<()>{
        if count>self.capacity_frames{return Err("video chunk exceeds configured capacity".into());}
        if let Some([w,h,frames])=self.request.delivery{
            if first>=frames{return Ok(());}
            count=count.min(frames-first);
            if (w,h)!=(width,height){
                if self.resize_scratch==0{check(unsafe{cuMemAlloc_v2(&mut self.resize_scratch,self.capacity_frames*height*w*3*4)})?;}
                if self.resize_output==0{check(unsafe{cuMemAlloc_v2(&mut self.resize_output,self.capacity_frames*h*w*3)})?;}
                check(unsafe{h3vg_delivery_resize(device as _,self.resize_scratch as _,self.resize_output as _,
                    count as _,height as _,width as _,h as _,w as _,arena.stream)})?;
                device=self.resize_output;
            }
        }
        if first!=self.next||count==0||first+count>self.frames{return Err("invalid finalized VAE frame range".into());}
        let slot=if let Some(slot)=self.available.pop(){slot}else{self.returned.recv().map_err(|_|"VAE stream writer failed")?};
        let bytes=count*self.pixels*3;let host=&self.buffers[slot];
        if let Some(health)=&self.health {if self.health_error.is_none(){
            if let Err(error)=health.enqueue(device,count,self.pixels,arena.stream){self.health_error=Some(error.to_string());}
        }}
        if let Some(file)=self.capture.as_mut(){host.download(device,bytes,arena)?;let data=host.bytes(bytes);file.write_all(data)?;if let Some(hash)=self.raw_hash.as_mut(){hash.update(data);}}
        let mut profile=self.emit_profile;
        let mut phase=std::time::Instant::now();
        let mut lap=|profile:&mut Option<[f64;3]>,i:usize|->Result<()>{
            if let Some(p)=profile.as_mut(){arena.sync()?;p[i]+=phase.elapsed().as_secs_f64();phase=std::time::Instant::now();}
            Ok(())};
        lap(&mut profile,0)?;
        lap(&mut profile,1)?;
        host.download(device,bytes,arena)?;
        if self.health_error.is_none(){if let Some(health)=self.health.as_mut(){health.collect(count,self.pixels);}}
        lap(&mut profile,2)?;
        self.emit_profile=profile;
        self.sender.as_ref().ok_or("VAE stream already completed")?.send(FramePacket::Chunk{slot,first,count,bytes}).map_err(|_|"VAE stream writer failed")?;
        self.next+=count;self.chunks+=1;Ok(())
    }
    fn finish(&mut self)->Result<Value>{
        self.sender.take().ok_or("VAE stream already finished")?.send(FramePacket::End).map_err(|_|"VAE stream writer failed")?;
        self.writer.take().ok_or("VAE stream writer missing")?.join().map_err(|_|"VAE stream writer panicked")?.map_err(|e|->Box<dyn std::error::Error>{e.into()})?;
        let sha=self.raw_hash.take().map(|h|format!("{:x}",h.finalize()));
        Ok(json!({"frames":self.next,"framed_rgb24":true,"pinned_buffers":3,"queue_capacity":2,"raw_capture":self.capture.is_some(),"raw_sha256":sha,
            "chunks":self.chunks,
            "emit_profile_seconds":self.emit_profile.map(|p|json!({"health_and_capture":p[0],"device_sync":p[1],"download_and_health_collect":p[2]})),
            "output_health":{"frame_means":self.health.as_ref().map(|v|&v.means),"error":self.health_error}}))
    }
}
impl Drop for FrameStream {fn drop(&mut self){self.sender.take();if let Some(writer)=self.writer.take(){let _=writer.join();}
    // The writer has exited, so nothing reads these chunks any more.
    HostChunk::recycle(std::mem::take(&mut self.buffers));
    for device in [self.resize_scratch,self.resize_output]{if device!=0{unsafe{cuMemFree_v2(device);}}}}}

#[cfg(test)]
mod stream_contract_tests {
    use super::*;
    #[test]
    fn public_rgb_stream_has_no_raw_capture_or_delivery(){
        let args=vec!["vae","decode","model","latents","output","768","768","124","--rgb-stream","/tmp/rgb.sock"]
            .into_iter().map(String::from).collect::<Vec<_>>();
        let (clean,request)=request_arguments(&args).unwrap();let request=request.unwrap();
        assert_eq!(clean.len(),8);assert!(!request.capture_raw&&request.delivery.is_none());
        let mut bad=args.clone();*bad.last_mut().unwrap()="relative.sock".into();assert!(request_arguments(&bad).is_err());
        bad=args.clone();bad[1]="encode".into();assert!(request_arguments(&bad).is_err());
        bad=args.clone();bad.push("extra".into());assert!(request_arguments(&bad).is_err());
    }
    #[test]
    fn request_context_requires_explicit_valid_video_contract(){
        let mut value=json!({"schema":"h3-native-request-context-v1","video_stream_path":"/tmp/h3-video.sock","capture_video_raw":true});
        let request=VideoRequest::parse(&value).unwrap();assert!(request.capture_raw);
        value["video_stream_path"]=json!("relative.sock");assert!(VideoRequest::parse(&value).is_err());
    }
    #[test]
    fn original_temporal_chunks_cover_every_output_frame_once(){
        for tokens in 3..=400{
            let temporal=geometry::temporal(tokens).unwrap();let mut next=0;
            for chunk in 0..temporal.chunks{
                let first=chunk*17;let count=if chunk+1==temporal.chunks{temporal.frames-first}else{17.min(temporal.frames-first)};
                assert_eq!(first,next);assert!((1..=22).contains(&count));next+=count;
            }
            assert_eq!(next,temporal.frames);
        }
    }
    #[test]
    fn coordinator_handoff_covers_every_tile_once_without_raising_the_maximum(){
        for world in [2usize,4,8]{for tiles in [1usize,2,3,7,8,9,16,28,49,64,99]{
            let active=world.min(tiles);let coordinator=coordinator_computes(tiles,active);
            let mut seen=vec![0usize;tiles];let mut most=0;
            if coordinator{let mine=(0..tiles).filter(|i|i%active==0).count();most=mine;for i in (0..tiles).filter(|i|i%active==0){seen[i]+=1;}}
            for rank in 1..active{let mut count=0;for tile in worker_tiles(rank,tiles,active,coordinator){seen[tile]+=1;count+=1;}most=most.max(count);}
            assert!(seen.iter().all(|&n|n==1),"world {world} tiles {tiles}");
            assert_eq!(most,tiles.div_ceil(active),"world {world} tiles {tiles}");
        }}
        assert!(!coordinator_computes(28,8));assert!(coordinator_computes(16,8));
    }
    #[test]
    fn sparse_worker_population_handles_fewer_and_uneven_tiles(){
        for world in [1usize,2,4,8]{for tiles in [1usize,2,3,7,8,9,16,28,49,64,99]{
            let active=world.min(tiles);let mut seen=vec![0usize;tiles];let mut loads=vec![];
            for rank in 0..active{let mut count=0;for tile in assigned_tiles(rank,tiles,active){seen[tile]+=1;count+=1;}loads.push(count);}
            assert!(seen.iter().all(|n|*n==1));assert!(loads.iter().all(|n|*n>0));assert!(loads.iter().max().unwrap()-loads.iter().min().unwrap()<=1);
        }}
    }
    #[test]
    fn rejects_invalid_world_without_initializing_cuda(){
        for value in ["0","3","16","abc","-1"]{assert!(TileTeam::parse_world(value).is_err());}
        for value in ["1","2","4","8"]{assert_eq!(TileTeam::parse_world(value).unwrap().to_string(),value);}
    }
    #[test]
    fn failed_worker_does_not_skip_peer_drain_or_shutdown(){
        use std::sync::atomic::{AtomicUsize,Ordering};
        let stopped=Arc::new(AtomicUsize::new(0));let mut workers=vec![];
        for rank in 1..4{
            let (tx,rx)=mpsc::channel();let (reply_tx,reply_rx)=mpsc::channel();let stopped=stopped.clone();
            let thread=thread::spawn(move||{while let Ok(command)=rx.recv(){match command{
                TileCommand::Run(_)=>{reply_tx.send(if rank==1{Err("injected tile failure".into())}else{Ok(json!({"rank":rank}))}).unwrap();},
                TileCommand::Finish=>{reply_tx.send(Ok(json!({"finished":true}))).unwrap();},
                TileCommand::Stop=>{stopped.fetch_add(1,Ordering::SeqCst);break;},_=>panic!("unexpected test prepare"),
            }}});
            workers.push(TileWorker{command:tx,reply:reply_rx,thread:Some(thread)});
        }
        let mut team=TileTeam{workers,world:4,active:3,identity:vec![],completed:0,coordinator:true,queued:false,queued_tiles:vec![]};
        let count=team.dispatch(0).unwrap();assert_eq!(count,2);assert!(team.drain(count).is_err());
        assert!(team.workers[1].reply.try_recv().is_err());team.finish().unwrap();drop(team);
        assert_eq!(stopped.load(Ordering::SeqCst),3);
    }

}
