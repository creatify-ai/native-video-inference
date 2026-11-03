// Qwen is idle after conditioning. Preserve its immutable parameters in pinned
// host memory; only its device storage and local gathered-layer caches move.
// Streams, vendor plans and the process survive, while IPC mappings must be
// closed collectively before freeing any rank's exported shard region.
struct HostParameter {name:String,allocation:usize,offset:usize,bytes:usize}
struct HostAllocation {index:usize,offset:usize,bytes:usize}
struct HostWeights {pointer:*mut c_void,bytes:usize,params:Vec<HostParameter>,allocations:Vec<HostAllocation>,shard_offset:usize,shard_bytes:usize,
    allocation_seconds:f64,copy_seconds:f64}
impl Drop for HostWeights {
    fn drop(&mut self){if !self.pointer.is_null(){unsafe{cudaFreeHost(self.pointer);}}}
}
impl HostWeights {
    fn at(&self,offset:usize)->*mut c_void {unsafe{self.pointer.cast::<u8>().add(offset).cast()}}
    fn capture(model:&Model)->Result<Self> {
        let owned:Vec<_>=model.arena.allocations.iter().map(|b|(b.ptr as usize,b.bytes)).collect();
        let views:Vec<_>=model.params.values().map(|b|(b.ptr as usize,b.bytes)).collect();
        let locations=crate::weight_loader::tensor_allocation_locations(&owned,&views)?;
        let params:Vec<_>=model.params.iter().zip(locations).map(|((name,buffer),(allocation,offset))|
            HostParameter{name:name.clone(),allocation,offset,bytes:buffer.bytes}).collect();
        let indices:std::collections::BTreeSet<_>=params.iter().map(|p|p.allocation).collect();
        let mut allocations=vec![];let mut bytes=0usize;
        for index in indices {
            let buffer=&model.arena.allocations[index];
            allocations.push(HostAllocation{index,offset:bytes,bytes:buffer.bytes});
            bytes=bytes.checked_add(buffer.bytes).ok_or("Qwen host bytes overflow")?;
        }
        let shard_offset=bytes;let shard_bytes=model.text.as_ref().map_or(0,|t|t.region.bytes);
        bytes=bytes.checked_add(shard_bytes).ok_or("Qwen host bytes overflow")?;
        let mut saved=Self{pointer:ptr::null_mut(),bytes,params,allocations,shard_offset,shard_bytes,allocation_seconds:0.0,copy_seconds:0.0};
        let allocated=Instant::now();
        cuda(unsafe{cudaHostAlloc(&mut saved.pointer,bytes.max(1),0)})?;
        saved.allocation_seconds=allocated.elapsed().as_secs_f64();let copying=Instant::now();
        // Even a failed async copy must finish before the host buffer is dropped.
        let copied=(||->Result<()> {
            for allocation in &saved.allocations {let buffer=model.arena.allocations[allocation.index];
                cuda(unsafe{cudaMemcpyAsync(saved.at(allocation.offset),buffer.ptr,allocation.bytes,2,model.arena.stream)})?;
            }
            if let Some(text)=&model.text {
                cuda(unsafe{cudaMemcpyAsync(saved.at(shard_offset),text.region.ptr,shard_bytes,2,model.arena.stream)})?;
            }
            Ok(())
        })();
        let synced=model.arena.sync();copied?;synced?;saved.copy_seconds=copying.elapsed().as_secs_f64();Ok(saved)
    }
}
fn restored_buffer(bytes:usize)->Result<Buffer> {
    DYNAMIC_ALLOCATIONS.fetch_add(1,Ordering::Relaxed);
    let mut pointer=ptr::null_mut();cuda(unsafe{cudaMalloc(&mut pointer,bytes.max(1))})?;
    Ok(Buffer{ptr:pointer,bytes})
}
impl ConditionerServer {
    fn move_weights(&mut self,args:&[String])->Result<Value> {
        if args.len()!=1 {return Err("prepare-offload/offload/restore takes no arguments".into());}
        let offload=args[0]=="offload";let started=Instant::now();
        let prepare=args[0]=="prepare-offload";
        let Some(model)=self.model.as_mut() else {return Err("Qwen model has not been loaded".into());};
        if model.arena.allocations.len()!=self.weights_end+2 {return Err("Qwen request scratch is still live".into());}
        let mut transferred=0usize;let mut released=0usize;let mut capture_seconds=0.0;
        if (offload || prepare) && !self.offloaded {
            if server_identity(model)?!=self.identity {return Err("Qwen identity changed before offload".into());}
            self.in_gpu=true;model.arena.sync()?;
            if let Some(text)=&model.text {for stream in &text.copies {cuda(unsafe{cudaStreamSynchronize(*stream)})?;}}
            if self.host_weights.is_none() {
                let capture=Instant::now();let saved=HostWeights::capture(model)?;
                transferred=saved.bytes;self.host_weights=Some(saved);capture_seconds=capture.elapsed().as_secs_f64();
            }
        }
        if offload && !self.offloaded {
            if let Some(text)=&mut model.text {
                let group=self.group.as_ref().ok_or("sharded Qwen group missing")?;
                collective(unsafe{h3_comm_ipc_close(group.comm,text.bases.as_mut_ptr(),group.world as c_int)})?;
                text.mapped=false;text.views.fill(ptr::null_mut());text.filled=[None;2];
                for buffer in std::iter::once(&mut text.region).chain(text.slots.iter_mut()) {
                    cuda(unsafe{cudaFree(buffer.ptr)})?;released+=buffer.bytes;buffer.ptr=ptr::null_mut();
                }
            }
            let saved=self.host_weights.as_ref().ok_or("Qwen host snapshot missing")?;
            // Parallel Weights contains aliases of the device parameter buffers.
            model.weights=None;
            // Immutable parameters can share a slab. Only arena allocations
            // own device storage; parameter views must never be freed.
            for allocation in &saved.allocations {
                let buffer=&mut model.arena.allocations[allocation.index];
                cuda(unsafe{cudaFree(buffer.ptr)})?;released+=buffer.bytes;buffer.ptr=ptr::null_mut();
            }
            for param in &saved.params {
                model.params.get_mut(&param.name).ok_or("Qwen parameter disappeared")?.ptr=ptr::null_mut();
            }
            self.offloaded=true;
        }else if !offload && !prepare && self.offloaded {
            self.in_gpu=true;
            let saved=self.host_weights.as_ref().ok_or("Qwen host snapshot missing")?;
            let slots=model.text.as_ref().map_or(0,|t|t.slots.iter().map(|b|b.bytes).sum::<usize>());
            let (mut free,mut total)=(0usize,0usize);cuda(unsafe{cudaMemGetInfo(&mut free,&mut total)})?;
            let short=free<saved.bytes+slots;
            if match &self.group {Some(group)=>group.any(short)?,None=>short} {
                return Err("not enough GPU memory to restore Qwen after video output".into());
            }
            // Assign each successful allocation immediately so failure cleanup
            // owns it. The controller discards the entire group on GPU errors.
            for allocation in &saved.allocations {
                let buffer=restored_buffer(allocation.bytes)?;
                model.arena.allocations[allocation.index]=buffer;
                cuda(unsafe{cudaMemcpyAsync(buffer.ptr,saved.at(allocation.offset),allocation.bytes,1,model.arena.stream)})?;
            }
            for param in &saved.params {
                let owned=model.arena.allocations[param.allocation];
                let buffer=Buffer{ptr:unsafe{owned.ptr.cast::<u8>().add(param.offset).cast()},bytes:param.bytes};
                model.params.insert(param.name.clone(),buffer);
            }
            if let Some(text)=&mut model.text {
                text.region=restored_buffer(saved.shard_bytes)?;
                cuda(unsafe{cudaMemcpyAsync(text.region.ptr,saved.at(saved.shard_offset),saved.shard_bytes,1,model.arena.stream)})?;
                for slot in &mut text.slots {*slot=restored_buffer(slot.bytes)?;}
                text.filled=[None;2];
            }
            model.arena.sync()?;
            if let (Some(group),Some(text))=(self.group.as_ref(),model.text.as_mut()){text.map(group)?;}
            model.weights=None;self.identity=server_identity(model)?;
            self.offloaded=false;self.gpu_restore_count+=1;transferred=saved.bytes;
        }
        self.in_gpu=false;
        Ok(json!({"component":"conditioner","model_load_count":1,"gpu_restore_count":self.gpu_restore_count,
            "weights_and_handles_identity_sha256":self.identity,"request_wall_seconds":started.elapsed().as_secs_f64(),
            "parallel":self.group.as_ref().map(|g|json!({"world":g.world,"rank":g.rank})),
            "weight_transfer":{"operation":args[0],"offloaded":self.offloaded,"host_pinned_bytes":self.host_weights.as_ref().map_or(0,|h|h.bytes),
                "transferred_bytes":transferred,"device_bytes_released":released,"capture_seconds":capture_seconds,
                "host_allocation_seconds":if (offload || prepare) && transferred>0{self.host_weights.as_ref().map_or(0.0,|h|h.allocation_seconds)}else{0.0},
                "device_to_host_seconds":if (offload || prepare) && transferred>0{self.host_weights.as_ref().map_or(0.0,|h|h.copy_seconds)}else{0.0},
                "seconds":started.elapsed().as_secs_f64(),"immutable_host_snapshot_reused":(offload || prepare) && transferred==0,
                "gather_slots_copied":false,"model_files_reread":false,"gpu_restore_count":self.gpu_restore_count}}))
    }
}
