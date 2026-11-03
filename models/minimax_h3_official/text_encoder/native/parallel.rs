// Sequence-parallel Qwen for a resident group of conditioner ranks
// (H3_QWEN_WORLD > 1), one process per GPU, each with the complete weights.
// The vision tower runs on each rank's four-row merge groups and the text
// decoder on its contiguous token rows. Around attention, q/k/v move so each
// rank attends whole sequences for its head group (text: 64/world query and
// 8/world KV heads; vision: 16/world heads per image), then outputs return to
// their row owners. Both moves are direct pulls from peers' memory (CUDA IPC,
// as the DiT's direct exchange) after a stream-ordered barrier, straight into
// the layouts attention and the next projection consume. Projections use
// split-K-free GEMMs. From PARALLEL_MIN_ROWS text rows the accepted single-GPU
// calls select no split-K either, so every element keeps its reduction order;
// head-partitioned SDPA is also exact (checked bitwise on B200). Shorter
// sequences run unchanged on rank 0.
unsafe extern "C" {
    fn cudaSetDevice(device:c_int)->c_int;
    fn h3_comm_error()->*const c_char;
    fn h3_comm_unique_id_size()->usize;
    fn h3_comm_get_unique_id(id:*mut c_void,bytes:usize)->c_int;
    fn h3_comm_create(device:c_int,rank:c_int,world:c_int,id:*const c_void,bytes:usize,
        stream:Stream,timeout_ms:u64,validate:c_int,out:*mut *mut c_void)->c_int;
    fn h3_comm_set_stream(comm:*mut c_void,stream:Stream)->c_int;
    fn h3_comm_any(comm:*mut c_void,value:c_int,result:*mut c_int)->c_int;
    fn h3_comm_destroy(comm:*mut c_void);
    fn h3_comm_allgatherv(comm:*mut c_void,send:*const c_void,send_elements:usize,recv:*mut c_void,
        recv_elements:usize,rows:*const usize,width:usize,dtype:c_int)->c_int;
    fn h3_comm_ipc_open(comm:*mut c_void,local:*const c_void,views:*mut *mut c_void,bases:*mut *mut c_void,count:c_int)->c_int;
    fn h3_comm_ipc_close(comm:*mut c_void,bases:*mut *mut c_void,count:c_int)->c_int;
    fn h3_comm_ipc_unmap(comm:*mut c_void,bases:*mut *mut c_void,count:c_int)->c_int;
    fn h3_comm_stream_barrier(comm:*mut c_void)->c_int;
    fn q3_parallel_error()->*const c_char;
    fn q3_gemm_create(workspace:usize,out:*mut *mut c_void)->c_int;
    fn q3_gemm_destroy(handle:*mut c_void);
    fn q3_gemm_clear(handle:*mut c_void)->c_int;
    fn q3_gemm_plan_count(handle:*mut c_void)->usize;
    fn q3_gemm_replaced_count(handle:*mut c_void)->usize;
    fn q3_gemm_forward_bias(handle:*mut c_void,x:*const c_void,w:*const c_void,bias:*const c_void,y:*mut c_void,
        rows:c_int,n:c_int,k:c_int,stream:Stream)->c_int;
    fn q3_cp_pack3(a:*const c_void,sa:i64,b:*const c_void,sb:i64,c:*const c_void,sc:i64,wa:c_int,wb:c_int,wc:c_int,
        out:*mut c_void,rows:i64,world:c_int,stream:Stream)->c_int;
    fn q3_cp_pull3(peers:*const *const c_void,starts:*const i64,world:c_int,rank:c_int,wa:c_int,wb:c_int,wc:c_int,
        a:*mut c_void,pa:i64,b:*mut c_void,pb:i64,c:*mut c_void,pc:i64,stream:Stream)->c_int;
    fn q3_cp_pull_rows(peers:*const *const c_void,world:c_int,start:i64,rows:i64,width:c_int,pitch:i64,step:i64,
        out:*mut c_void,stream:Stream)->c_int;
    fn cudaEventCreate(event:*mut *mut c_void)->c_int;
    fn cudaEventRecord(event:*mut c_void,stream:Stream)->c_int;
    fn cudaEventElapsedTime(ms:*mut f32,start:*mut c_void,end:*mut c_void)->c_int;
    fn cudaEventDestroy(event:*mut c_void)->c_int;
    fn q3_cp_attention_error()->*const c_char;
    fn q3_cp_attention_workspace(kind:c_int,world:c_int,sequence:i64,bytes:*mut usize)->c_int;
    fn q3_cp_attention_forward(kind:c_int,world:c_int,sequence:i64,q:*const c_void,k:*const c_void,v:*const c_void,
        out:*mut c_void,workspace:*mut c_void,bytes:usize,scale:f32,stream:Stream)->c_int;
    fn q3_cp_attention_plan_name(kind:c_int,world:c_int,sequence:i64)->*const c_char;
    fn q3_cp_attention_cleanup()->c_int;
    fn q3_cp_attention_cache_size()->usize;
    fn _exit(status:c_int)->!;
}

/// Shortest sequence decoded in parallel. Below 784 rows the accepted
/// down_proj selects split-K, which no row partition can reproduce.
pub(super) const PARALLEL_MIN_ROWS:usize=1024;
const GEMM_PLAN_LIMIT:usize=1024;

/// A failed collective leaves peers blocked: exit without CUDA destructors and
/// let the group supervisor reap every rank.
fn collective(code:c_int)->Result<()> {
    if code==0{return Ok(());}
    eprintln!("fatal conditioner collective {code}: {}",unsafe{CStr::from_ptr(h3_comm_error())}.to_string_lossy());
    unsafe{_exit(72)}
}
fn env_usize(name:&str,default:usize)->Result<usize> {
    match std::env::var(name) {Ok(v)=>v.parse().map_err(|_|format!("invalid {name}").into()),
        Err(std::env::VarError::NotPresent)=>Ok(default),Err(_)=>Err(format!("invalid {name}").into())}
}
pub fn write_unique_id(path:&Path)->Result<()> {
    use std::io::Write;
    let mut data=vec![0u8;unsafe{h3_comm_unique_id_size()}];
    collective(unsafe{h3_comm_get_unique_id(data.as_mut_ptr().cast(),data.len())})?;
    let mut file=fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&data)?;file.sync_all()?;Ok(())
}
fn balanced(total:usize,world:usize)->Vec<usize> {(0..world).map(|r|total/world+usize::from(r<total%world)).collect()}
fn starts(counts:&[usize])->Vec<i64> {std::iter::once(0).chain(counts.iter().scan(0,|sum,&c|{*sum+=c as i64;Some(*sum)})).collect()}
/// Every rank's rows of one vision call: whole four-row merge groups, balanced.
fn vision_counts(grids:&[(usize,usize)],world:usize)->Vec<usize> {
    balanced(grids.iter().map(|&(h,w)|h*w).sum::<usize>()/4,world).into_iter().map(|g|g*4).collect()
}
/// This rank's pixel rows of one vision call, as vision_parallel embeds them.
pub(super) fn vision_slice(grids:&[(usize,usize)],world:usize,rank:usize)->std::ops::Range<usize> {
    let counts=vision_counts(grids,world);let start=counts[..rank].iter().sum::<usize>();
    start..start+counts[rank]
}

#[derive(Clone,Copy,PartialEq,Eq,Debug)]
pub(super) enum Role {Single,Parallel,Idle}

/// Stream-ordered phase boundaries, read after the forward's own synchronization.
pub(super) struct Phases {events:Vec<*mut c_void>,stream:Stream}
impl Phases {
    pub(super) fn new(stream:Stream)->Self {Self{events:vec![],stream}}
    pub(super) fn mark(&mut self)->Result<()> {
        let mut event=ptr::null_mut();
        cuda(unsafe{cudaEventCreate(&mut event)})?;self.events.push(event);
        cuda(unsafe{cudaEventRecord(event,self.stream)})
    }
    pub(super) fn seconds(&self)->Result<Vec<f64>> {
        self.events.windows(2).map(|pair|{let mut ms=0f32;cuda(unsafe{cudaEventElapsedTime(&mut ms,pair[0],pair[1])})?;Ok(f64::from(ms)/1000.0)}).collect()
    }
}
impl Drop for Phases {fn drop(&mut self){for event in &self.events{unsafe{cudaEventDestroy(*event);}}}}

/// Opt-in (H3_QWEN_PROFILE=1) labeled stream marks: each segment, including any
/// GPU idle while the host waits, is charged to the label that opened it.
struct Profile {marks:Vec<(&'static str,*mut c_void)>}
impl Profile {
    fn mark(&mut self,label:&'static str,stream:Stream)->Result<()> {
        let mut event=ptr::null_mut();
        cuda(unsafe{cudaEventCreate(&mut event)})?;self.marks.push((label,event));
        cuda(unsafe{cudaEventRecord(event,stream)})
    }
    fn take(&mut self)->Result<Value> {
        let mut totals=std::collections::BTreeMap::<&str,f64>::new();
        for pair in self.marks.windows(2) {
            let mut ms=0f32;cuda(unsafe{cudaEventElapsedTime(&mut ms,pair[0].1,pair[1].1)})?;
            *totals.entry(pair[0].0).or_default()+=f64::from(ms)/1000.0;
        }
        for (_,event) in self.marks.drain(..) {unsafe{cudaEventDestroy(event);}}
        Ok(json!(totals))
    }
}
impl Drop for Profile {fn drop(&mut self){for (_,event) in &self.marks{unsafe{cudaEventDestroy(*event);}}}}

/// Process-resident exchange storage, exported to every peer (CUDA IPC): q/k/v
/// staging in pack3's destination-major layout, then the local-head attention
/// output. Its size follows from the request shape alone, so every rank grows
/// it at the same point; old mappings close collectively before the free.
struct Shared {base:*mut c_void,stage:usize,out:usize,views:[*mut c_void;8],bases:[*mut c_void;8]}

pub(super) struct Group {comm:*mut c_void,gemm:*mut c_void,pub(super) rank:usize,pub(super) world:usize,bound:Stream,
    profile:Option<Profile>,shared:Option<Shared>}
impl Group {
    /// Select this rank's device before any other CUDA call and join the group.
    pub(super) fn from_environment()->Result<Option<Self>> {
        let world=env_usize("H3_QWEN_WORLD",1)?;
        if world==1{return Ok(None);}
        if ![2,4,8].contains(&world){return Err("H3_QWEN_WORLD must be 1, 2, 4 or 8".into());}
        let rank=env_usize("H3_QWEN_RANK",world)?;let device=env_usize("H3_QWEN_DEVICE",rank)?;
        if rank>=world || device>i32::MAX as usize{return Err("invalid conditioner rank/device".into());}
        cuda(unsafe{cudaSetDevice(device as c_int)})?;
        let id=fs::read(std::env::var("H3_QWEN_UNIQUE_ID_FILE").map_err(|_|"missing H3_QWEN_UNIQUE_ID_FILE")?)?;
        if id.len()!=unsafe{h3_comm_unique_id_size()}{return Err("NCCL unique ID size mismatch".into());}
        let timeout=env_usize("H3_QWEN_TIMEOUT_MS",120000)?;
        if !(1000..=600000).contains(&timeout){return Err("H3_QWEN_TIMEOUT_MS must be 1000..600000".into());}
        let mut comm=ptr::null_mut();
        collective(unsafe{h3_comm_create(device as c_int,rank as c_int,world as c_int,id.as_ptr().cast(),id.len(),
            ptr::null_mut(),timeout as u64,0,&mut comm)})?;
        let mut gemm=ptr::null_mut();
        native_error(unsafe{q3_gemm_create(32<<20,&mut gemm)},q3_parallel_error)?;
        eprintln!("conditioner rank {rank}/{world} joined on device {device}");
        let profile=(std::env::var("H3_QWEN_PROFILE").as_deref()==Ok("1")).then(||Profile{marks:vec![]});
        Ok(Some(Self{comm,gemm,rank,world,bound:ptr::null_mut(),profile,shared:None}))
    }
    /// Collectives run on the model's compute stream once it exists.
    pub(super) fn bind(&mut self,stream:Stream)->Result<()> {
        if self.bound!=stream{collective(unsafe{h3_comm_set_stream(self.comm,stream)})?;self.bound=stream;}
        Ok(())
    }
    pub(super) fn role(&self,text_rows:usize)->Role {
        if text_rows>=PARALLEL_MIN_ROWS.max(self.world){Role::Parallel}else if self.rank==0{Role::Single}else{Role::Idle}
    }
    /// Identical decision on every rank; input checks before it are deterministic.
    pub(super) fn any(&self,value:bool)->Result<bool> {
        let mut result=0;
        collective(unsafe{h3_comm_any(self.comm,c_int::from(value),&mut result)})?;
        Ok(result!=0)
    }
    fn mark(&mut self,label:&'static str)->Result<()> {
        match &mut self.profile {Some(profile)=>profile.mark(label,self.bound),None=>Ok(())}
    }
    /// Segment totals since the last call, after the forward has synchronized.
    pub(super) fn take_profile(&mut self)->Result<Value> {
        match &mut self.profile {Some(profile)=>profile.take(),None=>Ok(Value::Null)}
    }
    pub(super) fn report(&self,role:Role,rows:usize)->Value {
        let counts=balanced(rows,self.world);
        json!({"world":self.world,"rank":self.rank,"mode":match role{Role::Parallel=>"sequence_parallel",Role::Single=>"single_rank",Role::Idle=>"idle"},
            "minimum_parallel_rows":PARALLEL_MIN_ROWS,"rank_rows":if role==Role::Parallel{json!(counts)}else{Value::Null},
            "exchange":"direct_ipc_pull","exchange_bytes":self.shared.as_ref().map_or(0,|s|s.stage+s.out),
            "gemm_plans":unsafe{q3_gemm_plan_count(self.gemm)},"gemm_plans_off_tiny_tile":unsafe{q3_gemm_replaced_count(self.gemm)},"head_attention_plans":unsafe{q3_cp_attention_cache_size()}})
    }
    /// Grow the exchange storage to at least `stage` + `out` bytes, on every rank alike.
    fn reserve(&mut self,stage:usize,out:usize)->Result<()> {
        let (have_stage,have_out)=self.shared.as_ref().map_or((0,0),|s|(s.stage,s.out));
        if stage<=have_stage && out<=have_out {return Ok(());}
        let round=|bytes:usize|(bytes.max(1)+255)&!255;
        let (stage,out)=(round(stage.max(have_stage)),round(out.max(have_out)));
        if let Some(mut old)=self.shared.take() {
            collective(unsafe{h3_comm_ipc_close(self.comm,old.bases.as_mut_ptr(),self.world as c_int)})?;
            cuda(unsafe{cudaFree(old.base)})?;
        }
        let mut base=ptr::null_mut();
        cuda(unsafe{cudaMalloc(&mut base,stage+out)})?;
        let mut shared=Shared{base,stage,out,views:[ptr::null_mut();8],bases:[ptr::null_mut();8]};
        collective(unsafe{h3_comm_ipc_open(self.comm,base,shared.views.as_mut_ptr(),shared.bases.as_mut_ptr(),self.world as c_int)})?;
        self.shared=Some(shared);
        Ok(())
    }
    /// Every rank's staging (offset 0) or attention output (offset `stage`).
    fn peers(&self,out:bool)->Result<[*const c_void;8]> {
        let shared=self.shared.as_ref().ok_or("exchange storage missing")?;
        let mut views=[ptr::null();8];
        for (view,peer) in views.iter_mut().zip(&shared.views[..self.world]) {
            *view=unsafe{peer.cast::<u8>().add(if out{shared.stage}else{0})}.cast_const().cast();
        }
        Ok(views)
    }
    fn local(&self,out:bool)->Result<Buffer> {
        let shared=self.shared.as_ref().ok_or("exchange storage missing")?;
        Ok(if out{Buffer{ptr:unsafe{shared.base.cast::<u8>().add(shared.stage).cast()},bytes:shared.out}}else{Buffer{ptr:shared.base,bytes:shared.stage}})
    }
    /// Every rank has finished all earlier stream work, including its reads of
    /// peers' storage, once this completes on its stream.
    fn barrier(&self)->Result<()> {collective(unsafe{h3_comm_stream_barrier(self.comm)})}
    fn pull3(&self,stage:&[*const c_void;8],starts:&[i64],widths:[usize;3],targets:[(*mut c_void,usize);3],stream:Stream)->Result<()> {
        native_error(unsafe{q3_cp_pull3(stage.as_ptr(),starts.as_ptr(),self.world as c_int,self.rank as c_int,
            widths[0] as c_int,widths[1] as c_int,widths[2] as c_int,targets[0].0,targets[0].1 as i64,
            targets[1].0,targets[1].1 as i64,targets[2].0,targets[2].1 as i64,stream)},q3_parallel_error)
    }
}
impl Drop for Group {
    fn drop(&mut self){unsafe{
        if let Some(mut shared)=self.shared.take(){h3_comm_ipc_unmap(self.comm,shared.bases.as_mut_ptr(),self.world as c_int);cudaFree(shared.base);}
        q3_gemm_destroy(self.gemm);h3_comm_destroy(self.comm);
    }}
}

/// Head-group text attention workspace in the persistent slot after vision's.
fn attach_parallel_text_workspace(model:&mut Model,weights_end:usize,world:usize)->Result<()> {
    let mut bytes=0;
    native_error(unsafe{q3_cp_attention_workspace(1,world as c_int,model.layout.sequence(1),&mut bytes)},q3_cp_attention_error)?;
    let buffer=model.arena.alloc(bytes)?;
    let last=model.arena.allocations.pop().ok_or("text workspace allocation missing")?;
    if weights_end+1>model.arena.allocations.len(){return Err("persistent allocation boundary changed".into());}
    model.arena.allocations.insert(weights_end+1,last);
    model.attention_workspace[1]=buffer;
    Ok(())
}
fn parallel_plan_name(kind:c_int,world:usize,rows:usize)->Result<String> {
    let name=unsafe{q3_cp_attention_plan_name(kind,world as c_int,rows as i64)};
    if name.is_null(){return Err("parallel attention plan name failed".into());}
    Ok(unsafe{CStr::from_ptr(name)}.to_string_lossy().into_owned())
}

/// A BF16 projection Y[rows,n] = X[rows,k] W[n,k]^T (+ bias) and a norm (RMS
/// without bias, LayerNorm with it), resolved once per model.
#[derive(Clone,Copy)] struct Linear {w:*const c_void,b:*const c_void,n:usize,k:usize}
#[derive(Clone,Copy)] struct Norm {w:*const c_void,b:*const c_void}
struct TextLayer {input:Norm,q:Linear,q_norm:Norm,k:Linear,k_norm:Norm,v:Linear,o:Linear,post:Norm,gate:Linear,up:Linear,down:Linear}
struct VisionLayer {norm1:Norm,qkv:Linear,proj:Linear,norm2:Norm,fc1:Linear,fc2:Linear}
struct Merger {norm:Norm,fc1:Linear,fc2:Linear}
/// Every parameter the parallel forward reads per layer: the hot loops make no
/// name lookups.
pub(super) struct Weights {text:Vec<TextLayer>,vision:Vec<VisionLayer>,deep:Vec<Merger>,merger:Merger}

pub(super) struct TextPlan {counts:Vec<usize>,start:usize,n:usize,tokens:[usize;2],firsts:[usize;2],views:Vec<Buffer>}

fn offset(buffer:Buffer,elements:usize)->*mut c_void {unsafe{buffer.ptr.cast::<u8>().add(elements*2).cast()}}

impl Model {
    fn linear_spec(&self,name:&str,n:usize,k:usize)->Result<Linear> {
        let weight=self.p(&format!("{name}.weight"))?;
        let spec=&self.index["weights"][format!("{name}.weight")];
        if shape(spec)?!=vec![n,k] || spec["dtype"]!="torch.bfloat16" {return Err(format!("parallel linear spec {name}").into());}
        let bias=self.params.get(&format!("{name}.bias")).map_or(ptr::null(),|b|b.ptr.cast_const());
        Ok(Linear{w:weight.ptr.cast_const(),b:bias,n,k})
    }
    /// A decoder projection. Sharded weights have no resident pointer: the
    /// forward takes each layer's gathered view (text_views).
    fn text_linear_spec(&self,layer:usize,projection:usize)->Result<Linear> {
        let (suffix,n,k)=crate::text_shards::PROJECTIONS[projection];
        let name=format!("language_model.layers.{layer}.{suffix}");
        if self.text.is_none(){return self.linear_spec(&name,n,k);}
        let bias=self.params.get(&format!("{name}.bias")).map_or(ptr::null(),|b|b.ptr.cast_const());
        Ok(Linear{w:ptr::null(),b:bias,n,k})
    }
    fn norm_spec(&self,name:&str,layer_norm:bool)->Result<Norm> {
        let w=self.p(&format!("{name}.weight"))?.ptr.cast_const();
        Ok(Norm{w,b:if layer_norm{self.p(&format!("{name}.bias"))?.ptr.cast_const()}else{ptr::null()}})
    }
    fn merger_spec(&self,name:&str)->Result<Merger> {
        Ok(Merger{norm:self.norm_spec(&format!("{name}.norm"),true)?,fc1:self.linear_spec(&format!("{name}.linear_fc1"),4608,4608)?,
            fc2:self.linear_spec(&format!("{name}.linear_fc2"),5120,4608)?})
    }
    pub(super) fn parallel_weights(&mut self)->Result<std::rc::Rc<Weights>> {
        if let Some(weights)=&self.weights {return Ok(weights.clone());}
        if self.observer.is_some(){return Err("a parallel conditioner rank takes no validation observer".into());}
        let text=(0..50).map(|layer|{let n=format!("language_model.layers.{layer}");Ok(TextLayer{
            input:self.norm_spec(&format!("{n}.input_layernorm"),false)?,q:self.text_linear_spec(layer,0)?,
            q_norm:self.norm_spec(&format!("{n}.self_attn.q_norm"),false)?,k:self.text_linear_spec(layer,1)?,
            k_norm:self.norm_spec(&format!("{n}.self_attn.k_norm"),false)?,v:self.text_linear_spec(layer,2)?,
            o:self.text_linear_spec(layer,3)?,post:self.norm_spec(&format!("{n}.post_attention_layernorm"),false)?,
            gate:self.text_linear_spec(layer,4)?,up:self.text_linear_spec(layer,5)?,
            down:self.text_linear_spec(layer,6)?})}).collect::<Result<Vec<_>>>()?;
        let vision=(0..27).map(|layer|{let n=format!("visual.blocks.{layer}");Ok(VisionLayer{
            norm1:self.norm_spec(&format!("{n}.norm1"),true)?,qkv:self.linear_spec(&format!("{n}.attn.qkv"),3456,1152)?,
            proj:self.linear_spec(&format!("{n}.attn.proj"),1152,1152)?,norm2:self.norm_spec(&format!("{n}.norm2"),true)?,
            fc1:self.linear_spec(&format!("{n}.mlp.linear_fc1"),4304,1152)?,fc2:self.linear_spec(&format!("{n}.mlp.linear_fc2"),1152,4304)?})})
            .collect::<Result<Vec<_>>>()?;
        let deep=(0..3).map(|i|self.merger_spec(&format!("visual.deepstack_merger_list.{i}"))).collect::<Result<Vec<_>>>()?;
        let weights=std::rc::Rc::new(Weights{text,vision,deep,merger:self.merger_spec("visual.merger")?});
        self.weights=Some(weights.clone());
        Ok(weights)
    }
    /// Linear on this rank's rows; a bias uses the accepted BIAS epilogue.
    fn gemm(&self,group:&Group,l:&Linear,x:Buffer,out:Buffer,rows:usize)->Result<()> {
        if rows==0{return Ok(());}
        native_error(unsafe{q3_gemm_forward_bias(group.gemm,x.ptr,l.w,l.b,out.ptr,rows as c_int,l.n as c_int,l.k as c_int,self.arena.stream)},q3_parallel_error)
    }
    fn norm_rows(&self,n:&Norm,x:Buffer,out:Buffer,rows:usize,cols:usize)->Result<()> {
        if n.b.is_null() {cuda(unsafe{q3_rms(x.ptr,n.w,out.ptr,rows as i64,cols as c_int,1e-6,self.arena.stream)})}
        else {cuda(unsafe{q3_layernorm(x.ptr,n.w,n.b,out.ptr,rows as i64,cols as c_int,1e-6,self.arena.stream)})}
    }
    fn activation(&self,x:Buffer,n:usize,kind:c_int)->Result<()> {
        cuda(unsafe{q3_activation(x.ptr,x.ptr,n as i64,kind,self.arena.stream)})
    }
    fn merger_rows(&self,group:&Group,m:&Merger,x:Buffer,norm:Buffer,fc:Buffer,out:Buffer,post:bool,rows:usize)->Result<()> {
        if rows==0{return Ok(());}
        let merged=rows/4;
        self.norm_rows(&m.norm,x,norm,if post{merged}else{rows},if post{4608}else{1152})?;
        self.gemm(group,&m.fc1,norm,fc,merged)?;
        self.activation(fc,merged*4608,2)?;
        self.gemm(group,&m.fc2,fc,out,merged)
    }
    /// vision_grids_file on this rank's merge groups (four rows each), from the
    /// rank's own pixel rows (vision_slice). Every rank returns all merged and
    /// deepstack rows. Ranks may own no rows.
    pub(super) fn vision_parallel(&mut self,group:&mut Group,grids:&[(usize,usize)],local:&[f32])->Result<(Buffer,[Buffer;3])> {
        let weights=self.parallel_weights()?;
        let world=group.world;let rank=group.rank;let stream=self.arena.stream;
        group.bind(stream)?;
        let rows=grids.iter().map(|&(h,w)|h*w).sum::<usize>();let merged_rows=rows/4;
        let max_sequence=grids.iter().map(|&(h,w)|h*w).max().ok_or("missing vision grids")?;
        let counts=vision_counts(grids,world);let groups:Vec<usize>=counts.iter().map(|c|c/4).collect();
        let starts=starts(&counts);
        let (start,n,m)=(starts[rank] as usize,counts[rank],groups[rank]);
        if local.len()!=n*1536 {return Err("rank pixel rows".into());}
        group.mark("vision.embed")?;
        let mut needed=0;
        for &(h,w) in grids {let mut size=0;native_error(unsafe{q3_cp_attention_workspace(0,world as c_int,(h*w) as i64,&mut size)},q3_cp_attention_error)?;needed=needed.max(size);}
        if self.attention_workspace[0].bytes<needed {self.attention_workspace[0]=self.arena.alloc(needed)?;}
        let ws=self.attention_workspace[0];
        let hw=1152/world;let b=2;
        group.reserve(counts.iter().max().copied().unwrap_or(0)*3456*b,rows*1152*b)?;
        let (stage,of)=(group.local(false)?,group.local(true)?);
        let (stage_peers,out_peers)=(group.peers(false)?,group.peers(true)?);
        let views=self.carve(&[n*1536*4,n*1536*b,rows*1536*b,rows*1152*b,rows*72*4,rows*72*4,rows*1152*b,
            2*max_sequence*72*4,2*max_sequence*72*4,2*max_sequence*1152*b,
            n*1152*b,n*1152*b,n*1152*b,n*3456*b,n*1152*b,n*1152*b,n*1152*b,n*4304*b,
            rows*1152*b,rows*1152*b,rows*3456*b,
            m*5120*b,m*5120*b,m*5120*b,m*5120*b,m*4608*b,
            merged_rows*5120*b,merged_rows*5120*b,merged_rows*5120*b,merged_rows*5120*b])?;
        let &[pixels,local_input,input,embedded,cos,sin,position,temp_cos,temp_sin,temp_position,
              hidden,norm,residual,qkv,q,k,attn,mlp,qf,kf,vf,
              deep0,deep1,deep2,merged,mergefc,full0,full1,full2,full3]=views.as_slice() else {return Err("vision plan views".into())};
        let deep=[deep0,deep1,deep2];
        // Each rank converts only its own pixel rows; NVLink then assembles the
        // accepted all-row convolution input on every rank.
        self.copy_in(pixels,unsafe{std::slice::from_raw_parts(local.as_ptr().cast::<u8>(),local.len()*4)})?;
        if n>0 {cuda(unsafe{q3_cast_bf16(pixels.ptr.cast(),local_input.ptr,(n*1536) as i64,stream)})?;}
        collective(unsafe{h3_comm_allgatherv(group.comm,local_input.ptr,n*1536,input.ptr,rows*1536,counts.as_ptr(),1536,2)})?;
        // The patch embedding keeps its accepted all-row convolution geometry.
        let spec=ConvSpec{ndim:5,dtype:2,transposed:0,allow_tf32:1,deterministic:0,channels_last:0,
            input_dims:[rows as i64,3,2,16,16],input_strides:[1536,512,256,16,1],
            weight_dims:[1152,3,2,16,16],weight_strides:[1536,512,256,16,1],
            output_dims:[rows as i64,1152,1,1,1],output_strides:[1152,1,1,1,1],padding:[0;3],stride:[2,16,16],dilation:[1;3]};
        let w=self.p("visual.patch_embed.proj.weight")?;let bias=self.p("visual.patch_embed.proj.bias")?;
        native_error(unsafe{h3_conv_forward(self.conv,&spec,input.ptr,w.ptr,embedded.ptr,stream)},h3_conv_error)?;
        self.conv_plan=unsafe{CStr::from_ptr(h3_conv_plan_tag(self.conv))}.to_string_lossy().into_owned();
        // Rotary and position tables for every row, exactly as vision_grids_file.
        let table=self.p("visual.pos_embed.weight")?;let inv=self.p("visual.rotary_pos_emb.inv_freq")?;
        let mut at=0;
        for &(h,w) in grids {
            let count=h*w;
            cuda(unsafe{q3_vision_tables(table.ptr,inv.ptr.cast(),temp_cos.ptr.cast(),temp_sin.ptr.cast(),temp_position.ptr,h as i32,w as i32,stream)})?;
            for (source,destination,stride) in [(temp_cos,cos,72*4),(temp_sin,sin,72*4),(temp_position,position,1152*2)] {
                let dest=destination.slice(at*stride,count*stride)?;
                cuda(unsafe{cudaMemcpyAsync(dest.ptr,source.ptr,count*stride,3,stream)})?;
            }
            at+=count;
        }
        let (cos,sin)=(cos.slice(start*288,n*288)?,sin.slice(start*288,n*288)?);
        if n>0 {
            cuda(unsafe{cudaMemcpyAsync(hidden.ptr,offset(embedded,start*1152),n*2304,3,stream)})?;
            self.add(hidden,bias,hidden,n*1152,1152)?;
            self.add(hidden,position.slice(start*2304,n*2304)?,hidden,n*1152,0)?;
        }
        // After the pull this rank attends all rows for heads [rank*h, rank*h+h)
        // in the accepted full-head row strides: q/k/out 1152 and v 3456.
        let scale=72f64.powf(-0.5) as f32;
        for (layer,w) in weights.vision.iter().enumerate() {
            group.mark("vision.projections")?;
            if n>0 {
                self.norm_rows(&w.norm1,hidden,norm,n,1152)?;
                self.gemm(group,&w.qkv,norm,qkv,n)?;
                cuda(unsafe{q3_vision_rope(qkv.ptr,cos.ptr.cast(),sin.ptr.cast(),q.ptr,k.ptr,n as i64,stream)})?;
                native_error(unsafe{q3_cp_pack3(q.ptr,1152,k.ptr,1152,offset(qkv,2304),3456,hw as c_int,hw as c_int,hw as c_int,
                    stage.ptr,n as i64,world as c_int,stream)},q3_parallel_error)?;
            }
            group.mark("vision.exchange")?;
            group.barrier()?;
            group.pull3(&stage_peers,&starts,[hw,hw,hw],[(offset(qf,rank*hw),1152),(offset(kf,rank*hw),1152),(offset(vf,2304+rank*hw),3456)],stream)?;
            group.mark("vision.attention")?;
            let mut image=0;
            for &(gh,gw) in grids {
                let sequence=gh*gw;
                native_error(unsafe{q3_cp_attention_forward(0,world as c_int,sequence as i64,offset(qf,image*1152+rank*hw),
                    offset(kf,image*1152+rank*hw),offset(vf,image*3456+2304+rank*hw),offset(of,image*1152+rank*hw),
                    ws.ptr,ws.bytes,scale,stream)},q3_cp_attention_error)?;
                image+=sequence;
            }
            group.mark("vision.exchange_back")?;
            group.barrier()?;
            if n>0 {native_error(unsafe{q3_cp_pull_rows(out_peers.as_ptr(),world as c_int,start as i64,n as i64,hw as c_int,1152,hw as i64,attn.ptr,stream)},q3_parallel_error)?;}
            group.mark("vision.mlp")?;
            if n>0 {
                self.gemm(group,&w.proj,attn,residual,n)?;
                self.add(hidden,residual,hidden,n*1152,0)?;
                self.norm_rows(&w.norm2,hidden,norm,n,1152)?;
                self.gemm(group,&w.fc1,norm,mlp,n)?;
                self.activation(mlp,n*4304,1)?;
                self.gemm(group,&w.fc2,mlp,residual,n)?;
                self.add(hidden,residual,hidden,n*1152,0)?;
                if let Some(index)=[8,16,24].iter().position(|&l|l==layer) {
                    self.merger_rows(group,&weights.deep[index],hidden,norm,mergefc,deep[index],true,n)?;
                }
            }
        }
        self.merger_rows(group,&weights.merger,hidden,norm,mergefc,merged,false,n)?;
        group.mark("vision.gather")?;
        for (local,gathered) in [(merged,full0),(deep[0],full1),(deep[1],full2),(deep[2],full3)] {
            collective(unsafe{h3_comm_allgatherv(group.comm,local.ptr,m*5120,gathered.ptr,merged_rows*5120,groups.as_ptr(),5120,2)})?;
        }
        group.mark("vision.end")?;
        Ok((full0,[full1,full2,full3]))
    }
    /// One device allocation for a request phase, split into 256-byte-aligned views.
    fn carve(&mut self,sizes:&[usize])->Result<Vec<Buffer>> {
        let aligned=|bytes:usize|(bytes.max(1)+255)&!255;
        let total=sizes.iter().try_fold(0usize,|sum,&bytes|sum.checked_add(aligned(bytes))).ok_or("scratch size overflow")?;
        let block=self.arena.alloc(total)?;
        let mut at=0;let mut views=Vec::with_capacity(sizes.len());
        for &bytes in sizes {views.push(block.slice(at,bytes)?);at+=aligned(bytes);}
        Ok(views)
    }
    /// Pageable upload: the host bytes are staged before this returns.
    fn copy_in(&self,target:Buffer,bytes:&[u8])->Result<()> {
        if target.bytes<bytes.len(){return Err("upload view too small".into());}
        if bytes.is_empty(){return Ok(());}
        cuda(unsafe{cudaMemcpyAsync(target.ptr,bytes.as_ptr().cast(),bytes.len(),1,self.arena.stream)})
    }
    /// Host inputs and one scratch allocation for text_parallel, prepared before
    /// the vision tower is queued so no upload waits on it mid-forward.
    pub(super) fn plan_text_parallel(&mut self,group:&Group,prepared:&Path)->Result<TextPlan> {
        let s=self.layout.text_rows;let world=group.world;
        let counts=balanced(s,world);let start:usize=counts[..group.rank].iter().sum();let n=counts[group.rank];
        if counts.contains(&0){return Err("parallel conditioner needs a row on every rank".into());}
        let idbytes=fs::read(prepared.join("input_ids.raw"))?;
        if idbytes.len()!=s*8 {return Err("unsupported native conditioner text length".into());}
        let positions=fs::read(prepared.join("position_ids.raw"))?;
        if positions.len()!=s*24 {return Err("prepared position byte count".into());}
        let config:Value=serde_json::from_slice(&fs::read(prepared.join("../model-config.json"))?)?;
        // Modality rows owned by this rank: image tokens are scanned in order, so
        // the j-th token takes feature row j, as in text_multimodal.
        let (mut tokens,mut firsts,mut local)=([0usize;2],[0usize;2],[vec![],vec![]]);
        for (m,key) in ["image_token_id","video_token_id"].into_iter().enumerate() {
            let id=config[key].as_i64().ok_or("modality token id")?;
            let indices:Vec<i64>=idbytes.chunks_exact(8).enumerate().filter_map(|(i,b)|(i64::from_le_bytes(b.try_into().unwrap())==id).then_some(i as i64)).collect();
            let first=indices.partition_point(|&i|(i as usize)<start);
            let last=indices.partition_point(|&i|(i as usize)<start+n);
            tokens[m]=indices.len();firsts[m]=first;
            local[m]=indices[first..last].iter().flat_map(|&i|(i-start as i64).to_le_bytes()).collect();
        }
        let b=2;
        let views=self.carve(&[n*8,s*24,local[0].len(),local[1].len(),
            n*5120*b,n*5120*b,n*5120*b,n*8192*b,n*8192*b,n*1024*b,n*1024*b,n*1024*b,n*8192*b,n*25600*b,n*25600*b,
            s*(8192/world)*b,s*(1024/world)*b,s*(1024/world)*b,s*5120*b,s*128*b,s*128*b])?;
        self.copy_in(views[0],&idbytes[start*8..(start+n)*8])?;
        self.copy_in(views[1],&positions)?;
        for m in 0..2 {self.copy_in(views[2+m],&local[m])?;}
        Ok(TextPlan{counts,start,n,tokens,firsts,views})
    }
    /// text_multimodal on this rank's rows; returns all rows' states on every rank.
    pub(super) fn text_parallel(&mut self,group:&mut Group,plan:TextPlan,vision:Option<(Buffer,[Buffer;3])>,video:Option<(Buffer,[Buffer;3])>)->Result<Buffer> {
        let weights=self.parallel_weights()?;
        let s=self.layout.text_rows;let world=group.world;let stream=self.arena.stream;
        group.bind(stream)?;
        group.mark("text.setup")?;
        if unsafe{q3_gemm_plan_count(group.gemm)}>GEMM_PLAN_LIMIT{native_error(unsafe{q3_gemm_clear(group.gemm)},q3_parallel_error)?;}
        let TextPlan{counts,start,n,tokens,firsts,views}=plan;
        let &[ids,positions,image_index,video_index,hidden,norm,residual,q,qnorm,k,knorm,v,attention,gate,up,
              qh,kh,vh,gathered,cos,sin]=views.as_slice() else {return Err("text plan views".into())};
        let mut insertions=vec![];
        for (m,(features,index)) in [(vision,image_index),(video,video_index)].into_iter().enumerate() {
            if features.map_or(0,|(v,_)|v.bytes/(5120*2))!=tokens[m] {return Err("vision token count".into());}
            let count=index.bytes/8;
            if let Some((features,deep))=features.filter(|_|count>0) {
                let (offset,bytes)=(firsts[m]*5120*2,count*5120*2);
                insertions.push((features.slice(offset,bytes)?,
                    [deep[0].slice(offset,bytes)?,deep[1].slice(offset,bytes)?,deep[2].slice(offset,bytes)?],index,count));
            }
        }
        let (qwidth,kvwidth)=(8192/world,1024/world);
        group.reserve(counts.iter().max().copied().unwrap_or(0)*10240*2,s*qwidth*2)?;
        let (stage,oh)=(group.local(false)?,group.local(true)?);
        let (stage_peers,out_peers)=(group.peers(false)?,group.peers(true)?);
        let starts=starts(&counts);
        let ws=self.attention_workspace[1];let scale=128f64.powf(-0.5) as f32;
        let table=self.p("language_model.embed_tokens.weight")?;
        cuda(unsafe{q3_embedding(ids.ptr.cast(),table.ptr,hidden.ptr,n as i64,5120,stream)})?;
        for &(features,_,indices,count) in &insertions {cuda(unsafe{q3_image_insert(hidden.ptr,features.ptr,indices.ptr.cast(),count as i32,0,stream)})?;}
        // Rotary tables for every row are cheap; each rank reads its own rows.
        let inv=self.p("language_model.rotary_emb.inv_freq")?;
        cuda(unsafe{q3_text_tables(positions.ptr.cast(),inv.ptr.cast(),cos.ptr,sin.ptr,s as i32,stream)})?;
        let (cos,sin)=(cos.slice(start*256,n*256)?,sin.slice(start*256,n*256)?);
        for (layer,w) in weights.text.iter().enumerate() {
            group.mark("text.projections")?;
            // A sharded rank reads this layer's projections from its gathered block.
            let views=self.text_views(layer)?;
            let at=|l:&Linear,i:usize|->Linear{match views{Some(v)=>Linear{w:v[i].ptr.cast_const(),..*l},None=>*l}};
            self.norm_rows(&w.input,hidden,norm,n,5120)?;
            self.gemm(group,&at(&w.q,0),norm,q,n)?;
            self.norm_rows(&w.q_norm,q,qnorm,n*64,128)?;
            self.gemm(group,&at(&w.k,1),norm,k,n)?;
            self.norm_rows(&w.k_norm,k,knorm,n*8,128)?;
            self.gemm(group,&at(&w.v,2),norm,v,n)?;
            cuda(unsafe{q3_text_rope(qnorm.ptr,cos.ptr,sin.ptr,q.ptr,n as i64,64,stream)})?;
            cuda(unsafe{q3_text_rope(knorm.ptr,cos.ptr,sin.ptr,k.ptr,n as i64,8,stream)})?;
            native_error(unsafe{q3_cp_pack3(q.ptr,8192,k.ptr,1024,v.ptr,1024,qwidth as c_int,kvwidth as c_int,kvwidth as c_int,
                stage.ptr,n as i64,world as c_int,stream)},q3_parallel_error)?;
            group.mark("text.exchange")?;
            group.barrier()?;
            group.pull3(&stage_peers,&starts,[qwidth,kvwidth,kvwidth],[(qh.ptr,qwidth),(kh.ptr,kvwidth),(vh.ptr,kvwidth)],stream)?;
            group.mark("text.attention")?;
            native_error(unsafe{q3_cp_attention_forward(1,world as c_int,s as i64,qh.ptr,kh.ptr,vh.ptr,oh.ptr,ws.ptr,ws.bytes,scale,stream)},q3_cp_attention_error)?;
            group.mark("text.exchange_back")?;
            group.barrier()?;
            native_error(unsafe{q3_cp_pull_rows(out_peers.as_ptr(),world as c_int,start as i64,n as i64,qwidth as c_int,qwidth as i64,0,attention.ptr,stream)},q3_parallel_error)?;
            group.mark("text.mlp")?;
            self.gemm(group,&at(&w.o,3),attention,residual,n)?;
            self.add(hidden,residual,hidden,n*5120,0)?;
            self.norm_rows(&w.post,hidden,norm,n,5120)?;
            self.gemm(group,&at(&w.gate,4),norm,gate,n)?;
            self.activation(gate,n*25600,0)?;
            self.gemm(group,&at(&w.up,5),norm,up,n)?;
            cuda(unsafe{q3_binary(gate.ptr,up.ptr,gate.ptr,(n*25600) as i64,1,0,stream)})?;
            self.gemm(group,&at(&w.down,6),gate,residual,n)?;
            self.text_layer_done(layer)?;
            self.add(hidden,residual,hidden,n*5120,0)?;
            if layer<3 {for &(_,deep,indices,count) in &insertions {cuda(unsafe{q3_image_insert(hidden.ptr,deep[layer].ptr,indices.ptr.cast(),count as i32,1,stream)})?;}}
        }
        group.mark("text.gather")?;
        collective(unsafe{h3_comm_allgatherv(group.comm,hidden.ptr,n*5120,gathered.ptr,s*5120,counts.as_ptr(),5120,2)})?;
        group.mark("text.end")?;
        Ok(gathered)
    }
}
