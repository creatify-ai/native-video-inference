//! Two-GPU sequence parallelism for the 48 DiT blocks.
//!
//! Each GPU runs every block on its own contiguous range of video tokens (the first GPU takes the
//! extra token of an odd count). Audio tokens and the text context are small and stay whole on both
//! GPUs, which compute them identically. Every operation of a block is independent per video token
//! except two attentions whose keys are video tokens: video self-attention and audio-to-video
//! attention. Before those two, each GPU gathers the other's keys and values, so its queries see all
//! of them. On B200 with the pinned libraries, GEMMs over a token range and cuDNN attention over a
//! query range or with gathered keys give the same bits as over all tokens, so the output stays
//! byte-identical to one GPU's.
//!
//! The per-GPU block program is the traced block program at the local token count. The profile
//! traces the block at stage 1's and stage 2's token counts, and every integer that differs between
//! the two (shapes, strides, offsets, view arguments) is that count times a constant; anything else
//! is rejected.
use crate::{cuda::{Device,Event,wait_event},tensor::{self,Tensor},Result};
use serde_json::{Map,Value};
use std::{cell::RefCell,collections::BTreeMap,path::{Path,PathBuf},rc::Rc,sync::mpsc};

/// A block program for one GPU's token range.
pub(crate) struct Local {
    pub graph:Value,
    /// Each external that holds video tokens, and its token axis. Other externals are whole.
    pub axes:BTreeMap<String,usize>,
}

const SDPA:&[&str]=&["aten._scaled_dot_product_cudnn_attention.default","aten.scaled_dot_product_attention.default"];

fn prefix(stage:usize)->String{format!("block.{stage}.0.0")}
fn video_tokens(graph:&Value,stage:usize)->Result<usize>{
    graph["externals"][format!("{}.kwargs.video.x",prefix(stage))]["shape"][1].as_u64()
        .map(|n|n as usize).ok_or_else(||"block program has no video token count".into())
}

/// `own` traced at `t_own` tokens and `other` at `t_other`: the same program at `tokens`.
fn rescale(own:&Value,other:&Value,t_own:i128,t_other:i128,tokens:i128,rename:&dyn Fn(&str)->String)->Result<Value>{
    Ok(match (own,other){
        (Value::Object(a),Value::Object(b))=>{
            if a.len()!=b.len(){return Err("stage block programs differ in structure".into());}
            let mut out=Map::new();
            for (key,value) in a{
                if key=="sha256"{continue;}  // the full-token activation's digest; never checked here
                let other=b.get(key).or_else(||b.get(&rename(key))).ok_or("stage block programs differ in fields")?;
                out.insert(key.clone(),rescale(value,other,t_own,t_other,tokens,rename)?);
            }
            Value::Object(out)
        }
        (Value::Array(a),Value::Array(b))=>{
            if a.len()!=b.len(){return Err("stage block programs differ in length".into());}
            Value::Array(a.iter().zip(b).map(|(x,y)|rescale(x,y,t_own,t_other,tokens,rename)).collect::<Result<_>>()?)
        }
        (Value::Number(x),Value::Number(y)) if x.is_i64() && y.is_i64()=>{
            let (x,y)=(x.as_i64().unwrap() as i128,y.as_i64().unwrap() as i128);
            if x==y{own.clone()}else{
                let slope=(x-y)/(t_own-t_other);
                if slope*(t_own-t_other)!=x-y || slope*t_own!=x{
                    return Err(format!("block program integer {x} is not proportional to the token count").into());
                }
                Value::from(i64::try_from(slope*tokens)?)
            }
        }
        _=>own.clone(),
    })
}

fn specs(graph:&Value)->Result<BTreeMap<String,Value>>{
    let mut map=BTreeMap::new();
    for (name,spec) in graph["externals"].as_object().ok_or("graph externals")?{map.insert(name.clone(),spec.clone());}
    fn visit(value:&Value,map:&mut BTreeMap<String,Value>){
        if let Some(name)=value.get("tensor").and_then(Value::as_str){map.insert(name.into(),value.clone());}
        else if let Some(values)=value.as_array(){for value in values{visit(value,map);}}
    }
    for node in graph["nodes"].as_array().ok_or("graph nodes")?{visit(&node["out"],&mut map);}
    Ok(map)
}

/// The block program of `stage` (`own`, traced at the full token count) at `tokens` local tokens.
pub(crate) fn localize(own:&Value,stage:usize,other:&Value,other_stage:usize,tokens:usize)->Result<Local>{
    let (t_own,t_other)=(video_tokens(own,stage)?,video_tokens(other,other_stage)?);
    if t_own==t_other{return Err("stage block programs share a token count".into());}
    let (own_prefix,other_prefix)=(prefix(stage),prefix(other_stage));
    let rename=|key:&str|key.replacen(&own_prefix,&other_prefix,1);
    let mut graph=rescale(own,other,t_own as i128,t_other as i128,tokens as i128,&rename)?;
    let (own_specs,other_specs)=(specs(own)?,specs(other)?);
    let tokens_axis=|own_spec:&Value,other_spec:&Value|->Result<Option<usize>>{
        let a=own_spec["shape"].as_array().ok_or("tensor shape")?;let b=other_spec["shape"].as_array().ok_or("tensor shape")?;
        let axes=(0..a.len()).filter(|&i|a[i]!=b[i]).collect::<Vec<_>>();
        match axes.as_slice(){[]=>Ok(None),[axis]=>Ok(Some(*axis)),_=>Err("tensor has more than one token axis".into())}
    };
    // Attentions whose keys are video tokens gather the other GPU's keys and values first.
    let other_nodes=other["nodes"].as_array().ok_or("graph nodes")?;
    for (index,node) in graph["nodes"].as_array_mut().ok_or("graph nodes")?.iter_mut().enumerate(){
        if !SDPA.contains(&node["op"].as_str().unwrap_or("")){continue;}
        let key=|graph_node:&Value,specs:&BTreeMap<String,Value>,arg:usize|->Result<Value>{
            let name=graph_node["args"][arg]["tensor"].as_str().ok_or("attention operand")?;
            specs.get(name).cloned().ok_or_else(||"undefined attention operand".into())
        };
        let own_node=&own["nodes"][index];let other_node=&other_nodes[index];
        let keys=tokens_axis(&key(own_node,&own_specs,1)?,&key(other_node,&other_specs,1)?)?;
        let values=tokens_axis(&key(own_node,&own_specs,2)?,&key(other_node,&other_specs,2)?)?;
        match (keys,values){
            (None,None)=>{}
            (Some(2),Some(2))=>{node["sp_gather"]=Value::Bool(true);}
            _=>return Err("attention keys and values split differently".into()),
        }
        if !node["args"].get(3).is_none_or(Value::is_null) || node["kwargs"].get("attn_mask").is_some_and(|v|!v.is_null()){
            return Err("sequence-parallel attention with a mask is not ported".into());
        }
    }
    let mut axes=BTreeMap::new();
    for (name,spec) in own["externals"].as_object().ok_or("graph externals")?{
        let other_spec=other["externals"].get(rename(name)).ok_or("stage block programs differ in externals")?;
        if let Some(axis)=tokens_axis(spec,other_spec)?{axes.insert(name.clone(),axis);}
    }
    if graph["outputs"].as_array().ok_or("graph outputs")?.is_empty(){return Err("block program has no outputs".into());}
    Ok(Local{graph,axes})
}

thread_local!{static LOCAL:RefCell<BTreeMap<(PathBuf,usize,usize),Rc<Local>>>=const{RefCell::new(BTreeMap::new())};}
/// `localize` for the profile at `root`, kept for later forwards of this thread.
pub(crate) fn local(root:&Path,stage:usize,own:&Value,tokens:usize)->Result<Rc<Local>>{
    let key=(root.to_path_buf(),stage,tokens);
    if let Some(value)=LOCAL.with(|c|c.borrow().get(&key).cloned()){return Ok(value);}
    let other_stage=1-stage;
    let other:Value=serde_json::from_slice(&std::fs::read(root.join(format!("{}.graph.json",prefix(other_stage))))?)?;
    let value=Rc::new(localize(own,stage,&other,other_stage,tokens)?);
    LOCAL.with(|c|c.borrow_mut().insert(key,value.clone()));
    Ok(value)
}

/// GPU `rank`'s token range of `total`: [start, start+count).
pub(crate) fn split(total:usize,rank:usize)->(usize,usize){
    let first=total.div_ceil(2);
    if rank==0{(0,first)}else{(first,total-first)}
}

/// `value`'s token range along `axis`, copied into `spec`'s layout.
pub(crate) fn narrow(value:&Tensor,axis:usize,(start,count):(usize,usize),spec:&Value,device:&Device)->Result<Tensor>{
    if axis>=value.shape.len() || start+count>value.shape[axis]{return Err("token range outside tensor".into());}
    let mut view=value.clone();
    view.shape[axis]=count;
    view.offset+=start*value.strides[axis]*tensor::dtype(&value.dtype)?.1;
    let list=|v:&Value|->Result<Vec<usize>>{v.as_array().ok_or("spec list")?.iter()
        .map(|n|n.as_u64().map(|n|n as usize).ok_or_else(||"spec extent".into())).collect()};
    let mut out=Tensor::allocate(list(&spec["shape"])?,list(&spec["stride"])?,
        spec["dtype"].as_str().ok_or("spec dtype")?.into(),device)?;
    out.reference_offset=spec["storage_offset"].as_u64().unwrap_or(0) as usize;
    if out.shape!=view.shape || out.dtype!=view.dtype{return Err("token range differs from its local layout".into());}
    tensor::copy(&view,&out,device.stream)?;
    Ok(out)
}

/// Published after a GPU has queued its local operands of one gather (and its own copy of them).
struct Ready {event:usize,sources:Vec<(u64,usize)>}
// Event handles and device addresses are plain values; each side only reads what the other published.
unsafe impl Send for Ready {}

/// One GPU's end of the key/value exchange of a forward. Both GPUs gather in the same order (the
/// same program), so the messages pair up; if one side fails, its end is dropped and the other
/// side's receive fails instead of waiting.
///
/// Each gather is one message each way. The other GPU pulls this GPU's local operands of gather g
/// after its stream waits for this GPU's event of gather g; it records its event of gather g+1
/// after that pull. So this GPU keeps its operands of gather g until its stream waits for the other
/// GPU's event of gather g+1; the last ones live until the forward ends (both GPUs synchronize).
pub(crate) struct Exchange {
    rank:usize,total:usize,contexts:[usize;2],
    send:mpsc::Sender<Ready>,receive:mpsc::Receiver<Ready>,
    /// Recorded events, kept until the forward is done (the other stream may still wait on them).
    events:RefCell<Vec<Event>>,
    /// The local operands of the last gather, until the other GPU has pulled them.
    held:RefCell<Vec<Tensor>>,
}
unsafe impl Send for Exchange {}
impl Exchange {
    /// Both ends for `total` tokens between the private contexts of GPU 0 and GPU 1.
    pub(crate) fn pair(total:usize,contexts:[usize;2])->(Self,Self){
        let (to_one,from_zero)=mpsc::channel();let (to_zero,from_one)=mpsc::channel();
        let end=|rank,send,receive|Self{rank,total,contexts,send,receive,events:RefCell::new(Vec::new()),held:RefCell::new(Vec::new())};
        (end(0,to_one,from_one),end(1,to_zero,from_zero))
    }
    pub(crate) fn rank(&self)->usize{self.rank}
    pub(crate) fn total(&self)->usize{self.total}

    /// Each local [1, heads, tokens, dim] view (of [1, tokens, heads, dim] storage) gathered with
    /// the other GPU's tokens into one [1, heads, total, dim] tensor of the same layout.
    pub(crate) fn gather(&self,locals:&[&Tensor],device:&Device)->Result<Vec<Tensor>>{
        let (start,count)=split(self.total,self.rank);let (peer_start,peer_count)=split(self.total,1-self.rank);
        let mut fulls=Vec::new();let mut sources=Vec::new();let mut rows=Vec::new();
        for local in locals{
            let (heads,dim)=(local.shape.get(1).copied().unwrap_or(0),local.shape.get(3).copied().unwrap_or(0));
            if local.shape.len()!=4 || local.shape[0]!=1 || local.shape[2]!=count
                || local.strides[1]!=dim || local.strides[2]!=heads*dim || local.strides[3]!=1{
                return Err("gathered attention operand is not a token-major local range".into());
            }
            let width=tensor::dtype(&local.dtype)?.1;let row=heads*dim*width;
            let full=Tensor::allocate(vec![1,heads,self.total,dim],vec![self.total*heads*dim,dim,heads*dim,1],
                local.dtype.clone(),device)?;
            let mut own=full.clone();own.shape[2]=count;own.offset+=start*row;
            tensor::copy(local,&own,device.stream)?;
            sources.push((local.ptr(),count*row));rows.push(row);fulls.push(full);
        }
        // The local operands (and this GPU's copy of them) are queued before this event.
        let ready=Event::record(device.stream)?;
        self.send.send(Ready{event:ready.handle(),sources}).map_err(|_|"the other GPU stopped during a sequence-parallel forward")?;
        self.events.borrow_mut().push(ready);
        let peer=self.receive.recv().map_err(|_|"the other GPU stopped during a sequence-parallel forward")?;
        if peer.sources.len()!=fulls.len(){return Err("sequence-parallel exchange operand count differs".into());}
        wait_event(device.stream,peer.event)?;
        // The other GPU recorded that event after pulling the previous gather's operands.
        *self.held.borrow_mut()=locals.iter().map(|t|(*t).clone()).collect();
        for ((full,(address,bytes)),row) in fulls.iter().zip(peer.sources).zip(rows){
            if bytes!=peer_count*row{return Err("sequence-parallel exchange extent differs".into());}
            unsafe{native_core_infra::device_context::copy_peer(full.ptr()+(peer_start*row) as u64,self.contexts[self.rank],
                address,self.contexts[1-self.rank],bytes,device.stream)}?;
        }
        Ok(fulls)
    }
}
impl crate::graph::SequenceGather for Exchange {
    fn gather(&self,locals:&[&Tensor],device:&Device)->Result<Vec<Tensor>>{Exchange::gather(self,locals,device)}
}
