// Device side of crate::text_shards: this rank's shard of every decoder layer,
// every peer's shards mapped once with CUDA IPC, and two rotating slots that
// each hold one gathered layer block. Shards never change after loading, so a
// rank copies its peers' shards without their participation (copy engines, one
// stream per source rank) and rank 0 can run a short request alone. Events order
// each slot's refill after the compute stream finished the layer that read it.
unsafe extern "C" {
    fn cudaStreamWaitEvent(stream:Stream,event:*mut c_void,flags:u32)->c_int;
    fn cudaEventCreateWithFlags(event:*mut *mut c_void,flags:u32)->c_int;
    fn cudaIpcCloseMemHandle(pointer:*mut c_void)->c_int;
}

struct TextShards {
    layout:crate::text_shards::Layout,region:Buffer,slots:[Buffer;2],
    views:[*mut c_void;8],bases:[*mut c_void;8],mapped:bool,
    copies:Vec<Stream>,done:[Vec<*mut c_void>;2],free:[*mut c_void;2],
    /// The layer each slot was last filled with.
    filled:[Option<usize>;2],
}
impl TextShards {
    fn new(layout:crate::text_shards::Layout)->Result<Self> {
        let malloc=|bytes:usize|->Result<Buffer>{
            DYNAMIC_ALLOCATIONS.fetch_add(1,Ordering::Relaxed);
            let mut pointer=ptr::null_mut();cuda(unsafe{cudaMalloc(&mut pointer,bytes)})?;Ok(Buffer{ptr:pointer,bytes})
        };
        let event=||->Result<*mut c_void>{let mut e=ptr::null_mut();cuda(unsafe{cudaEventCreateWithFlags(&mut e,2)})?;Ok(e)};
        let mut shards=Self{layout,region:Buffer{ptr:ptr::null_mut(),bytes:0},slots:[Buffer{ptr:ptr::null_mut(),bytes:0};2],
            views:[ptr::null_mut();8],bases:[ptr::null_mut();8],mapped:false,copies:vec![],done:[vec![],vec![]],
            free:[ptr::null_mut();2],filled:[None;2]};
        shards.region=malloc(crate::text_shards::LAYERS*layout.shard)?;
        for slot in 0..2 {shards.slots[slot]=malloc(layout.block)?;}
        for _ in 0..layout.world {
            let mut stream=ptr::null_mut();cuda(unsafe{cudaStreamCreateWithFlags(&mut stream,1)})?;shards.copies.push(stream);
            for slot in 0..2 {let e=event()?;shards.done[slot].push(e);}
        }
        for slot in 0..2 {shards.free[slot]=event()?;}
        Ok(shards)
    }
    /// This rank's shard of one layer, for loading.
    fn own(&self,layer:usize)->Result<Buffer> {self.region.slice(layer*self.layout.shard,self.layout.shard)}
    /// Map every rank's shards; a collective on the group's bound stream.
    fn map(&mut self,group:&Group)->Result<()> {
        if self.mapped || group.world!=self.layout.world || group.rank!=self.layout.rank {return Err("text shard group mismatch".into());}
        collective(unsafe{h3_comm_ipc_open(group.comm,self.region.ptr,self.views.as_mut_ptr(),self.bases.as_mut_ptr(),group.world as c_int)})?;
        self.mapped=true;Ok(())
    }
    /// Queue the copy of `layer` from every rank's shard into its slot.
    fn fill(&mut self,layer:usize)->Result<()> {
        if !self.mapped {return Err("text shards are not mapped".into());}
        let (slot,shard)=(layer%2,self.layout.shard);
        for (source,stream) in self.copies.iter().enumerate() {
            cuda(unsafe{cudaStreamWaitEvent(*stream,self.free[slot],0)})?;
            let from=unsafe{self.views[source].cast::<u8>().add(layer*shard)};
            let to=unsafe{self.slots[slot].ptr.cast::<u8>().add(source*shard)};
            cuda(unsafe{cudaMemcpyAsync(to.cast(),from.cast(),shard,4,*stream)})?;
            cuda(unsafe{cudaEventRecord(self.done[slot][source],*stream)})?;
        }
        self.filled[slot]=Some(layer);Ok(())
    }
    /// Start copying the first two layers, e.g. while the vision tower runs.
    fn prefetch(&mut self)->Result<()> {
        for layer in 0..2 {if self.filled[layer%2]!=Some(layer){self.fill(layer)?;}}
        Ok(())
    }
    /// Make `stream` wait for `layer`'s block and return its seven weight views.
    fn layer(&mut self,layer:usize,stream:Stream)->Result<[Buffer;7]> {
        if layer>=crate::text_shards::LAYERS {return Err("text layer out of range".into());}
        if self.filled[layer%2]!=Some(layer) {self.fill(layer)?;}
        let slot=self.slots[layer%2];
        for event in &self.done[layer%2] {cuda(unsafe{cudaStreamWaitEvent(stream,*event,0)})?;}
        let mut views=[Buffer{ptr:ptr::null_mut(),bytes:0};7];
        for (i,view) in views.iter_mut().enumerate() {*view=slot.slice(self.layout.offsets[i],self.layout.bytes[i])?;}
        Ok(views)
    }
    /// `stream` has queued every read of `layer`: refill its slot two layers on.
    fn done(&mut self,layer:usize,stream:Stream)->Result<()> {
        cuda(unsafe{cudaEventRecord(self.free[layer%2],stream)})?;
        if layer+2<crate::text_shards::LAYERS {self.fill(layer+2)?;}
        Ok(())
    }
    fn report(&self)->Value {
        json!({"world":self.layout.world,"rank":self.layout.rank,"block_bytes":self.layout.block,
            "shard_bytes_per_layer":self.layout.shard,"resident_bytes":self.layout.resident_bytes(),"gather":"copy_engine_peer_pull"})
    }
    fn identity(&self)->Value {
        json!({"region":format!("{:p}",self.region.ptr),"slots":self.slots.iter().map(|s|format!("{:p}",s.ptr)).collect::<Vec<_>>(),
            "views":self.views[..self.layout.world].iter().map(|v|format!("{:p}",*v)).collect::<Vec<_>>()})
    }
}
impl Drop for TextShards {
    fn drop(&mut self){unsafe{
        for stream in &self.copies {cudaStreamSynchronize(*stream);}
        for base in self.bases.iter().filter(|b|!b.is_null()) {cudaIpcCloseMemHandle(*base);}
        for event in self.done.iter().flatten().chain(&self.free).filter(|e|!e.is_null()) {cudaEventDestroy(*event);}
        for stream in &self.copies {cudaStreamDestroy(*stream);}
        for buffer in self.slots.iter().chain(std::iter::once(&self.region)).filter(|b|!b.ptr.is_null()) {cudaFree(buffer.ptr);}
    }}
}
