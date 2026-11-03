// Included inside native.rs so the distributed path uses the same checked
// buffers, vendor wrappers and model constants as the accepted single GPU path.
unsafe extern "C" {
    fn cudaSetDevice(device:c_int)->c_int;
    fn h3_comm_error()->*const c_char;
    fn h3_comm_unique_id_size()->usize;
    fn h3_comm_get_unique_id(id:*mut c_void,bytes:usize)->c_int;
    fn h3_comm_create(device:c_int,rank:c_int,world:c_int,id:*const c_void,bytes:usize,
        stream:Stream,timeout_ms:u64,validate:c_int,out:*mut *mut c_void)->c_int;
    fn h3_comm_set_stream(comm:*mut c_void,stream:Stream)->c_int;
    fn h3_comm_synchronize(comm:*mut c_void)->c_int;
    fn h3_comm_any(comm:*mut c_void,value:c_int,result:*mut c_int)->c_int;
    fn h3_comm_destroy(comm:*mut c_void);
    fn _exit(status:c_int)->!;
    fn h3_comm_alltoallv(comm:*mut c_void,send:*const c_void,send_elements:usize,
        send_counts:*const usize,send_offsets:*const usize,recv:*mut c_void,recv_elements:usize,
        recv_counts:*const usize,recv_offsets:*const usize,dtype:c_int)->c_int;
    fn h3_comm_allgatherv(comm:*mut c_void,send:*const c_void,send_elements:usize,
        recv:*mut c_void,recv_elements:usize,rows:*const usize,width:usize,dtype:c_int)->c_int;
    fn h3_cp_pack_qkv(q:*const c_void,k:*const c_void,v:*const c_void,out:*mut c_void,
        rows:i64,heads:c_int,world:c_int,stream:Stream)->c_int;
    fn h3_cp_unpack_qkv(input:*const c_void,q:*mut c_void,k:*mut c_void,v:*mut c_void,
        rows:i64,heads:c_int,stream:Stream)->c_int;
    fn h3_cp_merge_heads(input:*const c_void,out:*mut c_void,rows:i64,heads:c_int,
        world:c_int,stream:Stream)->c_int;
    fn h3_comm_ipc_open(comm:*mut c_void,local:*const c_void,views:*mut *mut c_void,bases:*mut *mut c_void,count:c_int)->c_int;
    fn h3_comm_ipc_unmap(comm:*mut c_void,bases:*mut *mut c_void,count:c_int)->c_int;
    fn h3_comm_stream_barrier(comm:*mut c_void)->c_int;
    fn h3_cp_pull_out(attn:*const *const c_void,world:c_int,start:i64,rows:i64,out:*mut c_void,stream:Stream)->c_int;
    fn h3_cp_pull_qkv_packed(send:*const *const c_void,starts:*const u32,world:c_int,rank:c_int,seq:i64,
        oq:*mut c_void,ok:*mut c_void,ov:*mut c_void,stream:Stream)->c_int;
}
/// Peer views of the four buffers the direct exchange reads: q, k, v after RoPE and
/// the local-head attention output. Pure relayout; every BF16 bit is copied unchanged.
/// Process-resident direct-exchange storage: the packed q/k/v staging and the
/// local-head attention output, sized for the configured packed-row bound and
/// exported to every peer once. Layouts borrow prefixes; mappings are never
/// reopened, so freed-and-reallocated device addresses can never alias a stale
/// peer mapping. Closed only at process teardown.
struct DirectResident {send:Buffer,attn:Buffer,views:[[*mut c_void;8];2],bases:[[*mut c_void;8];2],
    rows_capacity:usize,seq_capacity:usize,handle:*mut c_void,world:usize}
impl DirectResident {
    fn create(group:&CpGroup)->Result<Self> {
        let bound=std::env::var("H3_NATIVE_MAX_PACKED_ROWS").ok().map(|v|v.parse::<usize>().map_err(|_|"invalid H3_NATIVE_MAX_PACKED_ROWS"))
            .transpose()?.unwrap_or(131_072).min(crate::limits::MAX_PACKED_ROWS);
        let seq_capacity=bound;let rows_capacity=bound.div_ceil(group.world);let heads=HEADS/group.world;
        let alloc=|bytes:usize|->Result<Buffer>{let mut ptr=ptr::null_mut();cuda(unsafe{cudaMalloc(&mut ptr,bytes)})?;Ok(Buffer{ptr,bytes})};
        let send=alloc(rows_capacity.checked_mul(INNER*3*2).ok_or("direct staging overflow")?)?;
        let attn=match alloc(seq_capacity.checked_mul(heads*128*2).ok_or("direct attention overflow")?) {
            Ok(buffer)=>buffer,Err(error)=>{unsafe{cudaFree(send.ptr);}return Err(error);}};
        let mut resident=Self{send,attn,views:[[ptr::null_mut();8];2],bases:[[ptr::null_mut();8];2],rows_capacity,seq_capacity,handle:group.handle,world:group.world};
        for (i,buffer) in [send,attn].into_iter().enumerate() {
            cp_check(unsafe{h3_comm_ipc_open(group.handle,buffer.ptr,resident.views[i].as_mut_ptr(),resident.bases[i].as_mut_ptr(),group.world as c_int)})?;
        }
        Ok(resident)
    }
}
impl Drop for DirectResident {
    fn drop(&mut self) {
        for bases in self.bases.iter_mut() { let _=unsafe{h3_comm_ipc_unmap(self.handle,bases.as_mut_ptr(),self.world as c_int)}; }
        unsafe{cudaFree(self.send.ptr);cudaFree(self.attn.ptr);}
    }
}
/// Per-layout view of the resident exchange storage plus this layout's row starts.
struct DirectExchange {resident:std::rc::Rc<DirectResident>,starts:Vec<u32>}
fn cp_check(code:c_int)->Result<()> {
    if code==0{return Ok(());}
    // A collective failure is not a recoverable request error. Do not enter
    // CUDA allocation destructors, which could wait on a missing peer. The
    // native coordinator notices process exit and kills/reaps the entire group.
    eprintln!("fatal native collective {code}: {}",unsafe{CStr::from_ptr(h3_comm_error())}.to_string_lossy());
    unsafe{_exit(72)}
}
fn cp_env_usize(name:&str,default:usize)->Result<usize> {
    match std::env::var(name) {Ok(value)=>value.parse().map_err(|_|format!("invalid {name}").into()),
        Err(std::env::VarError::NotPresent)=>Ok(default),Err(_)=>Err(format!("invalid {name}").into())}
}
pub fn cp_select_device_from_environment()->Result<()> {
    let world=cp_env_usize("H3_CP_WORLD",1)?;let rank=cp_env_usize("H3_CP_RANK",0)?;
    let device=cp_env_usize("H3_CP_DEVICE",rank)?;
    if ![1,2,4,8].contains(&world)||rank>=world||device>i32::MAX as usize{return Err("invalid CP rank/world/device".into());}
    // Leave the accepted world=1 path's device selection untouched unless explicit.
    if world>1||std::env::var_os("H3_CP_DEVICE").is_some(){cuda(unsafe{cudaSetDevice(device as c_int)})?;}
    Ok(())
}
pub fn cp_write_unique_id(path:&Path)->Result<()> {
    use std::io::Write;
    let mut data=vec![0u8;unsafe{h3_comm_unique_id_size()}];
    cp_check(unsafe{h3_comm_get_unique_id(data.as_mut_ptr().cast(),data.len())})?;
    let mut file=fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&data)?;file.sync_all()?;Ok(())
}
struct CpGroup {handle:*mut c_void,rank:usize,world:usize,
    direct:std::cell::RefCell<Option<std::rc::Rc<DirectResident>>>}
impl CpGroup {
    /// The resident direct-exchange storage, created on first use with the
    /// communicator bound to the requesting layout's stream (collective).
    fn direct_resident(&self)->Result<std::rc::Rc<DirectResident>> {
        if let Some(existing)=self.direct.borrow().as_ref() {return Ok(existing.clone());}
        let created=std::rc::Rc::new(DirectResident::create(self)?);
        *self.direct.borrow_mut()=Some(created.clone());Ok(created)
    }
}
impl CpGroup {
    fn from_environment()->Result<Option<std::rc::Rc<Self>>> {
        cp_select_device_from_environment()?;
        let world=cp_env_usize("H3_CP_WORLD",1)?;
        if world==1{return Ok(None);}
        let rank=cp_env_usize("H3_CP_RANK",0)?;let device=cp_env_usize("H3_CP_DEVICE",rank)?;
        let id=crate::startup_trace::read(std::env::var("H3_CP_UNIQUE_ID_FILE").map_err(|_|"missing H3_CP_UNIQUE_ID_FILE")?)?;
        if id.len()!=unsafe{h3_comm_unique_id_size()}{return Err("NCCL unique ID size mismatch".into());}
        let timeout=cp_env_usize("H3_CP_TIMEOUT_MS",120000)?;
        if !(1000..=600000).contains(&timeout){return Err("CP timeout must be 1000..600000 ms".into());}
        let validate=cp_env_usize("H3_CP_VALIDATE_COLLECTIVES",0)?;
        if validate>1{return Err("H3_CP_VALIDATE_COLLECTIVES must be 0 or 1".into());}
        let mut handle=ptr::null_mut();
        let _trace=crate::startup_trace::Span::new("dit_nccl_init","",0);
        cp_check(unsafe{h3_comm_create(device as c_int,rank as c_int,world as c_int,id.as_ptr().cast(),
            id.len(),ptr::null_mut(),timeout as u64,validate as c_int,&mut handle)})?;
        Ok(Some(std::rc::Rc::new(Self{handle,rank,world,direct:std::cell::RefCell::new(None)})))
    }
    fn bind(&self,stream:Stream)->Result<()> {cp_check(unsafe{h3_comm_set_stream(self.handle,stream)})}
    fn exchange(&self,send:Buffer,sc:&[usize],recv:Buffer,rc:&[usize])->Result<()> {
        let offsets=|counts:&[usize]|->Result<Vec<usize>>{
            let mut total=0usize;let mut result=Vec::with_capacity(counts.len());
            for count in counts{result.push(total);total=total.checked_add(*count).ok_or("CP count overflow")?;}Ok(result)
        };
        if sc.len()!=self.world||rc.len()!=self.world{return Err("CP peer counts mismatch".into());}
        let so=offsets(sc)?;let ro=offsets(rc)?;
        cp_check(unsafe{h3_comm_alltoallv(self.handle,send.ptr,send.bytes/2,sc.as_ptr(),so.as_ptr(),
            recv.ptr,recv.bytes/2,rc.as_ptr(),ro.as_ptr(),2)})
    }
}
impl Drop for CpGroup {fn drop(&mut self){unsafe{h3_comm_destroy(self.handle);}}}
/// A communicator formed on a helper thread while the owner thread uploads its weight shards;
/// only the owner thread wraps and uses it, after `join`.
pub(crate) struct PendingGroup {thread:Option<std::thread::JoinHandle<std::result::Result<usize,String>>>,rank:usize,world:usize}
impl PendingGroup {
    pub(crate) fn start()->Result<Option<Self>> {
        let world=cp_env_usize("H3_CP_WORLD",1)?;
        if world==1{return Ok(None);}
        let rank=cp_env_usize("H3_CP_RANK",0)?;
        let thread=std::thread::Builder::new().name("dit-nccl-init".into()).spawn(move||->std::result::Result<usize,String> {
            let group=CpGroup::from_environment().map_err(|e|e.to_string())?.ok_or("CP world changed")?;
            let group=std::rc::Rc::try_unwrap(group).map_err(|_|"new communicator is shared")?;
            // The owner thread takes over the handle; this thread never destroys it.
            let handle=group.handle as usize;std::mem::forget(group);Ok(handle)
        })?;
        Ok(Some(Self{thread:Some(thread),rank,world}))
    }
    fn join(mut self)->Result<std::rc::Rc<CpGroup>> {
        let _trace=crate::startup_trace::Span::new("dit_nccl_join","",0);
        let thread=self.thread.take().ok_or("communicator already joined")?;
        let handle=thread.join().map_err(|_|"communicator thread panicked")??;
        Ok(std::rc::Rc::new(CpGroup{handle:handle as *mut c_void,rank:self.rank,world:self.world,direct:std::cell::RefCell::new(None)}))
    }
}
impl Drop for PendingGroup {fn drop(&mut self){if let Some(thread)=self.thread.take(){let _=thread.join();}}}

struct CpState {
    group:std::rc::Rc<CpGroup>,counts:Vec<usize>,start:usize,rows:usize,heads:usize,
    send:Option<Buffer>,receive:Option<Buffer>,q:Buffer,k:Buffer,v:Buffer,attn:Buffer,returned:Option<Buffer>,
    dense_workspace:Buffer,bsa_workspace:Buffer,
    router:std::rc::Rc<Program>,route_buffers:HashMap<String,u64>,
    direct:Option<DirectExchange>,
}
impl CpState {
    fn new(arena:&mut Arena,model:&ResidentModel,router_root:&Path,seq:usize)->Result<Option<Self>> {
        let group=match &model.cp_group{Some(group)=>group.clone(),None=>return Ok(None)};
        if seq<group.world{return Err("CP requires at least one packed row per rank".into());}
        let counts=crate::scratch_plan::row_counts(seq,group.world)?;
        let rows=counts[group.rank];let start=counts[..group.rank].iter().sum();let heads=HEADS/group.world;
        let bytes=seq.checked_mul(heads*128*2).ok_or("CP allocation overflow")?;
        let q=arena.sparse_input(seq,heads*128)?;let k=arena.sparse_input(seq,heads*128)?;let v=arena.sparse_input(seq,heads*128)?;
        let exchange=std::env::var("H3_CP_EXCHANGE").unwrap_or_else(|_|"nccl".into());
        if !["nccl","direct"].contains(&exchange.as_str()){return Err("H3_CP_EXCHANGE must be nccl or direct".into());}
        // NCCL staging is only needed by the all-to-all path; direct pulls read peers in place.
        let staging=exchange=="nccl";
        let max_rows=*counts.iter().max().ok_or("empty CP counts")?;
        group.bind(arena.stream)?;
        let resident=if staging {None} else {Some(group.direct_resident()?)};
        let (attn,send)=match &resident {
            None=>(arena.alloc(bytes)?,Some(arena.alloc(max_rows*INNER*3*2)?)),
            Some(r)=>{
                if seq>r.seq_capacity || rows>r.rows_capacity {return Err("CP direct exchange capacity exceeded; raise H3_NATIVE_MAX_PACKED_ROWS".into());}
                (r.attn.prefix(bytes)?,Some(r.send.prefix(rows*INNER*3*2)?))
            }
        };
        let receive=if staging{Some(arena.alloc(bytes*3)?)}else{None};
        let returned=if staging{Some(arena.alloc(max_rows*INNER*2)?)}else{None};
        let dense_workspace=arena.alloc(model.dense.bytes_for_heads(seq,heads,0)?.max(1))?;
        let router=model.routers.get(router_root,seq,heads)?;
        let mut route_buffers=HashMap::new();
        for spec in &router.buffers {
            // As with one GPU, the router reads the logical Q/K rows; their storage keeps the
            // SubBlock guard rows after them.
            let memory=match spec.name.as_str(){"q"=>q,"k"=>k,_=>arena.alloc(spec.bytes)?};
            let logical=if matches!(spec.name.as_str(),"q"|"k"){bytes}else{memory.bytes};
            if logical!=spec.bytes{return Err("CP router buffer size mismatch".into());}
            route_buffers.insert(spec.name.clone(),memory.ptr as u64);
        }
        let mut bsa_bytes=0;
        cuda(unsafe{h3_bsa_masked_workspace_bytes(seq as c_int,heads as c_int,&mut bsa_bytes)})?;
        let bsa_workspace=arena.alloc(bsa_bytes.max(1))?;
        arena.cp_watchdog=Some(group.clone());
        let direct=match resident {
            None=>None,
            Some(resident)=>{
                let mut starts=vec![0u32];let mut total=0usize;
                for count in &counts{total+=count;starts.push(u32::try_from(total).map_err(|_|"CP row start overflow")?);}
                Some(DirectExchange{resident,starts})
            }
        };
        Ok(Some(Self{group,counts,start,rows,heads,send,receive,q,k,v,attn,returned,
            dense_workspace,bsa_workspace,router,route_buffers,direct}))
    }
}
impl Drop for CpState {
    fn drop(&mut self) {
        // Runtime places this field before Arena: detach while the borrowed stream lives.
        // Direct-exchange storage is process-resident; nothing is unmapped here.
        if let Err(error)=self.group.bind(ptr::null_mut()){eprintln!("CP stream detach failed: {error}");}
    }
}

impl Runtime {
    /// Group barrier on the compute stream (one-element NCCL allreduce).
    fn cp_barrier(&self,cp:&CpState)->Result<()> {
        cp_check(unsafe{h3_comm_stream_barrier(cp.group.handle)})
    }
    fn cp_diagnostic_boundary(&self,trace:&mut Option<Vec<Value>>,name:&str,
        buffer:Buffer,shape:&[usize])->Result<()> {
        let entries=match trace.as_mut(){Some(entries)=>entries,None=>return Ok(())};
        let elements=shape.iter().try_fold(1usize,|n,&v|n.checked_mul(v)).ok_or("CP trace shape overflow")?;
        let bytes=elements.checked_mul(2).ok_or("CP trace byte overflow")?;
        let data=self.arena.download(buffer.prefix(bytes)?)?;
        let rows=shape[0].min(64);let prefix=&data[..rows*(bytes/shape[0])];
        let mut prefix_shape=shape.to_vec();prefix_shape[0]=rows;
        let step=std::env::var("H3_NATIVE_TRACE_STEP").ok().and_then(|v|v.parse::<usize>().ok()).unwrap_or(0);
        let directory=self.output.join(format!("diagnostics/cp_step{step:02}_block00_rank00"));
        fs::create_dir_all(&directory)?;
        let file=format!("{name}.prefix64.raw");fs::write(directory.join(&file),prefix)?;
        entries.push(json!({"name":name,"dtype":"torch.bfloat16","shape":shape,
            "layout":"contiguous row-major","bytes":bytes,"sha256":crate::integrity::digest(&data),
            "full_tensor_hashed":true,"rank":0,"step":step,"layer":0,
            "raw_prefix":{"file":file,"shape":prefix_shape,"bytes":prefix.len(),
                "sha256":crate::integrity::digest(prefix)}}));
        fs::write(directory.join("trace.json"),serde_json::to_vec_pretty(entries)?)?;
        Ok(())
    }
    fn cp_attention(&self,cp:&CpState,name:&str,input:Buffer,sparse:bool,
        trace:&mut Option<Vec<Value>>)->Result<Buffer> {
        let rows=cp.rows;
        let(q,k,v,qn,kn)=(self.s("q"),self.s("k"),self.s("v"),self.s("qn"),self.s("kn"));
        if self.manifest["modules"][name]["fused_projections"]==true {
            self.linear(&format!("{name}.to_qkv"),input,self.s("ff2"),rows)?;
            cuda(unsafe{h3_qkv_split_bf16(self.s("ff2").ptr,q.ptr,k.ptr,v.ptr,rows as i64,INNER as c_int,self.arena.stream)})?;
        }else{
            self.linear_qkv(name,input,q,k,v,rows)?;
        }
        for (label,buffer) in [("attn.to_q",q),("attn.to_k",k),("attn.to_v",v)] {
            self.cp_diagnostic_boundary(trace,label,buffer,&[rows,INNER])?;
        }
        let qname=format!("{name}.norm_q");let kname=format!("{name}.norm_k");
        let qw=self.w(&format!("{qname}.weight"))?;let kw=self.w(&format!("{kname}.weight"))?;
        for weight in [&qw,&kw]{if weight.spec.dtype!="torch.bfloat16"||weight.spec.shape!=[128]{return Err("CP QK norm weight mismatch".into());}}
        let cos=self.cos.slice(cp.start*96*self.numerics.rope_bytes(),rows*96*self.numerics.rope_bytes())?;
        let sin=self.sin.slice(cp.start*96*self.numerics.rope_bytes(),rows*96*self.numerics.rope_bytes())?;
        cuda(unsafe{(if self.numerics.sglang(){h3_sglang_qk_rope_v1}else if self.numerics.fused(){h3_prod_qk_rope}else{h3_qk_rope_opt_dispatch_v1})(q.ptr,k.ptr,qw.buffer.ptr,kw.buffer.ptr,cos.ptr,sin.ptr,
            qn.ptr,kn.ptr,q.ptr,k.ptr,rows as i64,HEADS as c_int,self.eps(&qname)?,self.eps(&kname)?,self.arena.stream)})?;
        self.cp_diagnostic_boundary(trace,"q_after_rope",q,&[rows,INNER])?;
        self.cp_diagnostic_boundary(trace,"k_after_rope",k,&[rows,INNER])?;
        if let Some(d)=&cp.direct {
            let _span=GpuSpan::new("cp.qkv_exchange",self.arena.stream)?;
            {
                // Destination-major staging makes every peer's slice one contiguous block.
                let _pack=GpuSpan::new("cp.qkv_exchange.pack",self.arena.stream)?;
                cuda(unsafe{h3_cp_pack_qkv(q.ptr,k.ptr,v.ptr,cp.send.ok_or("CP staging missing")?.ptr,rows as i64,HEADS as c_int,cp.group.world as c_int,self.arena.stream)})?;
            }
            {
                // All peers' staging is complete once every rank passes this point.
                let _barrier=GpuSpan::new("cp.qkv_exchange.barrier",self.arena.stream)?;
                self.cp_barrier(cp)?;
            }
            let _pull=GpuSpan::new("cp.qkv_exchange.pull",self.arena.stream)?;
            cuda(unsafe{h3_cp_pull_qkv_packed(d.resident.views[0].as_ptr().cast(),d.starts.as_ptr(),cp.group.world as c_int,
                cp.group.rank as c_int,self.seq as i64,cp.q.ptr,cp.k.ptr,cp.v.ptr,self.arena.stream)})?;
        } else {
            let _span=GpuSpan::new("cp.qkv_exchange",self.arena.stream)?;
            {
                let _pack=GpuSpan::new("cp.qkv_exchange.pack",self.arena.stream)?;
                cuda(unsafe{h3_cp_pack_qkv(q.ptr,k.ptr,v.ptr,cp.send.ok_or("CP staging missing")?.ptr,rows as i64,HEADS as c_int,cp.group.world as c_int,self.arena.stream)})?;
            }
            let width=cp.heads*128*3;
            {
                // Per-rank interval of the collective alone: transfer plus any wait for peers.
                let _alltoall=GpuSpan::new("cp.qkv_exchange.alltoall",self.arena.stream)?;
                cp.group.exchange(cp.send.ok_or("CP staging missing")?,&vec![rows*width;cp.group.world],cp.receive.ok_or("CP staging missing")?,
                    &cp.counts.iter().map(|r|r*width).collect::<Vec<_>>())?;
            }
            let _unpack=GpuSpan::new("cp.qkv_exchange.unpack",self.arena.stream)?;
            cuda(unsafe{h3_cp_unpack_qkv(cp.receive.ok_or("CP staging missing")?.ptr,cp.q.ptr,cp.k.ptr,cp.v.ptr,self.seq as i64,cp.heads as c_int,self.arena.stream)})?;
        }
        let scale=1.0f32/128.0f32.sqrt();
        if sparse {
            let _span=GpuSpan::new("cp.sparse_attention",self.arena.stream)?;
            cp.router.launch(&cp.route_buffers,self.arena.stream)?;
            cuda(unsafe{h3_bsa_forward_masked(cp.q.ptr,cp.k.ptr,cp.v.ptr,cp.route_buffers["indices"] as *const i32,
                self.block_sizes.ptr.cast(),self.query_block_mask.ptr.cast(),cp.attn.ptr,cp.bsa_workspace.ptr,cp.bsa_workspace.bytes,
                self.seq as c_int,cp.heads as c_int,cp.router.plan.kept as c_int,scale,self.arena.stream)})?;
            BSA_CPP_CALLS.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
            self.correct_dense_audio(cp.q,cp.k,cp.v,cp.attn,cp.heads)?;
        }else{
            let _span=GpuSpan::new("cp.dense_attention",self.arena.stream)?;
            self.dense.run_for_heads(cp.q,cp.k,cp.v,cp.attn,cp.dense_workspace,self.seq,cp.heads,0,scale,self.arena.stream)?;
        }
        self.cp_diagnostic_boundary(trace,"attention_local_heads",cp.attn,&[self.seq,cp.heads,128])?;
        if let Some(d)=&cp.direct {
            let _span=GpuSpan::new("cp.output_exchange",self.arena.stream)?;
            {
                let _barrier=GpuSpan::new("cp.output_exchange.barrier",self.arena.stream)?;
                self.cp_barrier(cp)?;
            }
            let _pull=GpuSpan::new("cp.output_exchange.pull",self.arena.stream)?;
            cuda(unsafe{h3_cp_pull_out(d.resident.views[1].as_ptr().cast(),cp.group.world as c_int,cp.start as i64,rows as i64,
                self.s("attn").ptr,self.arena.stream)})?;
        } else {
            let _span=GpuSpan::new("cp.output_exchange",self.arena.stream)?;
            let width=cp.heads*128;
            {
                let _alltoall=GpuSpan::new("cp.output_exchange.alltoall",self.arena.stream)?;
                cp.group.exchange(cp.attn,&cp.counts.iter().map(|r|r*width).collect::<Vec<_>>(),
                    cp.returned.ok_or("CP staging missing")?,&vec![rows*width;cp.group.world])?;
            }
            let _merge=GpuSpan::new("cp.output_exchange.merge",self.arena.stream)?;
            cuda(unsafe{h3_cp_merge_heads(cp.returned.ok_or("CP staging missing")?.ptr,self.s("attn").ptr,rows as i64,
                HEADS as c_int,cp.group.world as c_int,self.arena.stream)})?;
        }
        self.cp_diagnostic_boundary(trace,"attention_merged",self.s("attn"),&[rows,INNER])?;
        self.linear(&format!("{name}.to_out.0"),self.s("attn"),self.s("branch"),rows)?;
        self.cp_diagnostic_boundary(trace,"attn.to_out.0",self.s("branch"),&[rows,HIDDEN])?;
        Ok(self.s("branch"))
    }
    fn cp_block(&self,cp:&CpState,layer:usize,step:usize)->Result<()> {
        let rows=cp.rows;let name=format!("transformer_blocks.{layer}");
        // Observation only: no new kernel or collective and no gold activation reads.
        let trace_step=std::env::var("H3_NATIVE_TRACE_STEP").ok().and_then(|v|v.parse().ok()).unwrap_or(0);
        let mut trace=if layer==0&&step==trace_step&&cp.group.rank==0&&crate::debug_audit::enabled()
            &&std::env::var("H3_NATIVE_TRACE_CP_BLOCK0").as_deref()==Ok("1") {Some(Vec::new())} else {None};
        let hidden=self.s("hidden").slice(cp.start*HIDDEN*2,rows*HIDDEN*2)?;
        self.cp_diagnostic_boundary(&mut trace,"packed_input",hidden,&[rows,HIDDEN])?;
        let index=self.steps[step].adaln_indices.slice(cp.start*8,rows*8)?;
        let [shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp]=self.modulation(layer,step)?;
        let n1=format!("{name}.norm1");let n2=format!("{name}.norm2");
        let w1=self.w(&format!("{n1}.weight"))?;let w2=self.w(&format!("{n2}.weight"))?;
        for weight in [&w1,&w2]{if weight.spec.dtype!="torch.bfloat16"||weight.spec.shape!=[HIDDEN]{return Err("CP norm weight mismatch".into());}}
        cuda(unsafe{(if self.numerics.fused(){h3_prod_rms_mod_bf16}else{h3_pw_rms_mod_bf16})(hidden.ptr,w1.buffer.ptr,scale_msa.ptr,shift_msa.ptr,index.ptr.cast(),
            self.s("norm").ptr,rows as i64,HIDDEN as c_int,(6*HIDDEN) as c_int,self.eps(&n1)?,self.arena.stream)})?;
        self.cp_diagnostic_boundary(&mut trace,"norm1_modulated",self.s("norm"),&[rows,HIDDEN])?;
        let branch=self.cp_attention(cp,&format!("{name}.attn"),self.s("norm"),self.execution.sparse_layer(step,self.seq,layer)?,&mut trace)?;
        cuda(unsafe{(if self.numerics.fused(){h3_prod_gated_rms_mod_bf16}else{h3_pw_gated_rms_mod_bf16})(hidden.ptr,branch.ptr,gate_msa.ptr,w2.buffer.ptr,
            scale_mlp.ptr,shift_mlp.ptr,index.ptr.cast(),self.s("norm").ptr,rows as i64,HIDDEN as c_int,
            (6*HIDDEN) as c_int,self.eps(&n2)?,self.arena.stream)})?;
        self.cp_diagnostic_boundary(&mut trace,"hidden_after_attention",hidden,&[rows,HIDDEN])?;
        self.cp_diagnostic_boundary(&mut trace,"norm2_modulated",self.s("norm"),&[rows,HIDDEN])?;
        let branch=self.feedforward(&format!("{name}.ff"),self.s("norm"),rows,&mut trace)?;
        cuda(unsafe{h3_pw_gated_bf16(hidden.ptr,branch.ptr,gate_mlp.ptr,index.ptr.cast(),hidden.ptr,
            rows as i64,HIDDEN as c_int,(6*HIDDEN) as c_int,self.arena.stream)})?;
        self.cp_diagnostic_boundary(&mut trace,"block_output",hidden,&[rows,HIDDEN])
    }
    fn cp_gather_hidden(&self)->Result<()> {
        if let Some(cp)=&self.cp {
            let _span=GpuSpan::new("cp.hidden_gather",self.arena.stream)?;
            let hidden=self.s("hidden");
            let local=hidden.slice(cp.start*HIDDEN*2,cp.rows*HIDDEN*2)?;
            // Called only after a complete transformer block: norm is dead
            // until the next block or final normalization. Its full-sequence
            // BF16 storage has exactly the gather extent. Keep the original
            // disjoint collective and copy, ordered on the same compute stream,
            // without another S*HIDDEN*2 allocation (~3.8 GiB at 2K/345 frames).
            // This also applies to intermediate diagnostic gathers; block
            // diagnostics have already downloaded their live norm values.
            let gather=self.s("norm");
            if gather.bytes!=hidden.bytes {return Err("CP hidden gather scratch extent mismatch".into());}
            cp_check(unsafe{h3_comm_allgatherv(cp.group.handle,local.ptr,local.bytes/2,
                gather.ptr,gather.bytes/2,cp.counts.as_ptr(),HIDDEN,2)})?;
            self.arena.copy(hidden,gather)?;
        }Ok(())
    }
    /// Diagnostic report for explicitly pinned linear shapes.
    pub fn linear_pinned_report(&self)->Value {
        unsafe extern "C" { fn h3_linear_pinned_report(handle:*mut c_void,out:*mut c_char,size:c_int)->c_int; }
        let mut buffer=vec![0u8;1<<16];
        let code=unsafe{h3_linear_pinned_report(self.linear,buffer.as_mut_ptr().cast(),buffer.len() as c_int)};
        if code!=0 {return json!({"error":code});}
        let text=CStr::from_bytes_until_nul(&buffer).map(|s|s.to_string_lossy().into_owned()).unwrap_or_default();
        serde_json::from_str(&text).unwrap_or(Value::Null)
    }
    pub fn cp_report(&self)->Value {
        match &self.cp {None=>json!({"world":1,"enabled":false}),Some(cp)=>json!({
            "enabled":true,"world":cp.group.world,"rank":cp.group.rank,"local_rows":cp.rows,
            "row_start":cp.start,"rows_by_rank":cp.counts,"local_heads":cp.heads,
            "scheme":"sequence rows; all-to-all to complete sequence/local heads for attention",
            "exchange":if cp.direct.is_some(){"direct_ipc_pull"}else{"nccl_alltoallv"},
            "direct_resident_rows_capacity":cp.direct.as_ref().map(|d|d.resident.rows_capacity),
            "row_counts":cp.counts,
            "weights_replicated":self.resident_model.sharded.is_none(),"request_allocations_full_size":false,
            "projection_scratch_rows":self.s("q").bytes/(INNER*2),"full_head_router_allocated":false,
            "small_base_projection_full_m":self.seq<4096,
            "floating_collective_reductions":false,"independent_numerical_acceptance":"external gate required"})}
    }
}
