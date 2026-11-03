use crate::{cuda::{check, Buffer, Device, Stream}, Result};
use std::{ffi::{c_int, c_void}, rc::Rc};

pub fn dtype(value: &str) -> Result<(i32,usize)> {
    Ok(match value {"torch.bfloat16"=>(0,2),"torch.float32"=>(1,4),"torch.int64"=>(2,8),
        "torch.uint8"|"torch.bool"=>(3,1),"torch.int32"=>(4,4),"torch.float16"=>(5,2),other=>return Err(format!("unported dtype {other}").into())})
}
#[derive(Clone)]
pub struct Tensor {
    pub storage: Rc<Buffer>, pub offset: usize, pub shape: Vec<usize>, pub strides: Vec<usize>,
    pub dtype: String, pub reference_offset: usize,
}
#[repr(C)]
#[derive(Clone,Copy)]
pub struct View {data:*mut c_void, dtype:c_int, rank:c_int, dims:[i64;8], strides:[i64;8]}
fn poison_activations()->bool{
    static POISON:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
    *POISON.get_or_init(||std::env::var("LTX25_POISON").as_deref()==Ok("1"))
}
impl Tensor {
    pub fn allocate(shape:Vec<usize>,strides:Vec<usize>,dtype_name:String,device:&Device)->Result<Self> {
        if shape.len()!=strides.len() || shape.len()>8 {return Err("invalid tensor layout".into());}
        let span=if shape.contains(&0){0}else{shape.iter().zip(&strides).try_fold(1usize,|n,(&d,&s)|
            (d-1).checked_mul(s).and_then(|x|n.checked_add(x)).ok_or("tensor layout overflow"))?};
        let bytes=span.checked_mul(dtype(&dtype_name)?.1).ok_or("tensor bytes overflow")?;
        // Activations start as torch.empty's do; LTX25_POISON=1 fills them with 0xFF so an unwritten
        // output can never match a reference (a development check that rewrites every allocation).
        let storage=if poison_activations(){Buffer::poisoned(bytes,device)?}else{device.transient(bytes)?};
        Ok(Self{storage:Rc::new(storage),offset:0,shape,strides,dtype:dtype_name,reference_offset:0})
    }
    pub fn contiguous_shape(shape:Vec<usize>,dtype_name:String,device:&Device)->Result<Self>{
        let strides=contiguous_strides(&shape); Self::allocate(shape,strides,dtype_name,device)
    }
    pub fn elements(&self)->Result<usize>{self.shape.iter().try_fold(1usize,|n,&d|n.checked_mul(d).ok_or_else(||"tensor overflow".into()))}
    pub fn is_contiguous(&self)->bool{
        let mut expected=1;
        for (&size,&stride) in self.shape.iter().zip(&self.strides).rev(){
            if size>1 && stride!=expected{return false;}
            expected*=size.max(1);
        }
        true
    }
    pub fn ptr(&self)->u64{self.storage.at(self.offset)}
    pub fn view(&self)->Result<View>{
        let mut view=View{data:self.ptr() as _,dtype:dtype(&self.dtype)?.0,rank:self.shape.len().try_into()?,dims:[1;8],strides:[0;8]};
        for i in 0..self.shape.len(){view.dims[i]=self.shape[i].try_into()?;view.strides[i]=self.strides[i].try_into()?;}
        Ok(view)
    }
    pub fn contiguous(&self,device:&Device)->Result<Self>{
        let output=Self::contiguous_shape(self.shape.clone(),self.dtype.clone(),device)?;
        copy(self,&output,device.stream)?;Ok(output)
    }
    pub fn download(&self,device:&Device)->Result<Vec<u8>>{
        let value=self.contiguous(device)?; value.storage.download(device)
    }
}
pub fn contiguous_strides(shape:&[usize])->Vec<usize>{
    let mut result=vec![0;shape.len()];let mut stride=1;
    for i in (0..shape.len()).rev(){result[i]=stride;stride*=shape[i].max(1);}result
}
extern "C" {
    fn ltx25_tensor_spatial(x:View,y:View,pads:*const i64,mode:c_int,value:f32,n:i64,stream:Stream)->c_int;
    fn ltx25_tensor_index(op:c_int,x:View,index:View,y:View,dim:c_int,n:i64,stream:Stream)->c_int;
    fn ltx25_tensor_binary_sort(x:View,y:View,descending:c_int,stream:Stream)->c_int;
    fn ltx25_mean(x:*const c_void,y:*mut c_void,dtype:c_int,rank:c_int,shape:*const i64,
        strides:*const i64,mask:u32,stream:Stream)->c_int;
    fn ltx25_norm2(x:*const c_void,y:*mut c_void,rank:c_int,shape:*const i64,
        strides:*const i64,mask:u32,stream:Stream)->c_int;
    fn ltx25_tensor_copy(x:View,y:View,n:i64,stream:Stream)->c_int;
    fn ltx25_tensor_fill(y:View,n:i64,start:f32,step:f32,stream:Stream)->c_int;
    fn ltx25_tensor_elementwise(op:c_int,a:View,b:View,c:View,y:View,n:i64,scalar:f32,
        alpha:f32,scalar_b:c_int,stream:Stream)->c_int;
    fn ltx25_fused_elementwise(inputs:*const View,nin:c_int,outputs:*const View,out_regs:*const c_int,nout:c_int,
        ops:*const FuseOp,nops:c_int,n:i64,stream:Stream)->c_int;
    fn ltx25_tensor_rms(x:*const c_void,w:*const c_void,y:*mut c_void,rstd:*mut f32,
        rows:i64,cols:c_int,eps:f32,dtype:c_int,stream:Stream)->c_int;
    fn ltx25_tensor_layer_norm(x:*const c_void,w:*const c_void,bias:*const c_void,y:*mut c_void,
        mean:*mut f32,rstd:*mut f32,rows:i64,cols:c_int,eps:f32,dtype:c_int,stream:Stream)->c_int;
    fn ltx25_group_norm(x:*const c_void,w:*const c_void,bias:*const c_void,y:*mut c_void,stats:*mut c_void,
        batch:i64,channels:c_int,spatial:i64,groups:c_int,eps:f32,dtype:c_int,stream:Stream)->c_int;
}
pub fn spatial(x:&Tensor,y:&Tensor,pads:&[i64],mode:i32,value:f32,device:&Device)->Result<()>{
    if x.dtype!=y.dtype || x.shape.len()!=y.shape.len() || pads.len()%2!=0 || pads.len()>x.shape.len()*2{
        return Err("padding layout differs".into());
    }
    let mut before=[0i64;8];
    for i in 0..pads.len()/2{
        let axis=x.shape.len()-1-i;before[axis]=pads[2*i];
        if x.shape[axis] as i64 + pads[2*i]+pads[2*i+1]!=y.shape[axis] as i64{return Err("padding extent differs".into());}
    }
    check(unsafe{ltx25_tensor_spatial(x.view()?,y.view()?,before.as_ptr(),mode,value,
        y.elements()?.try_into()?,device.stream)},"spatial pad/nearest copy")
}
pub fn index(op:i32,x:&Tensor,indices:Option<&Tensor>,y:&Tensor,dim:usize,device:&Device)->Result<()>{
    if x.dtype!=y.dtype || dim>=y.shape.len(){return Err("indexing dtype/rank differs".into());}
    if let Some(indices)=indices{
        if indices.dtype!="torch.int64" || indices.shape!=y.shape || x.shape.len()!=y.shape.len(){
            return Err("gather shape/index dtype differs".into());
        }
    }
    check(unsafe{ltx25_tensor_index(op,x.view()?,indices.unwrap_or(x).view()?,y.view()?,dim.try_into()?,
        y.elements()?.try_into()?,device.stream)},"tensor gather/repeat")
}
pub fn binary_sort(x:&Tensor,y:&Tensor,descending:bool,device:&Device)->Result<()>{
    if x.shape.len()!=2 || x.shape!=y.shape || x.dtype!="torch.int32" || y.dtype!="torch.int64"
        || y.strides!=contiguous_strides(&y.shape) || x.shape[1]>4096{
        return Err("only stable binary-mask argsort along the last axis has been ported".into());
    }
    check(unsafe{ltx25_tensor_binary_sort(x.view()?,y.view()?,descending.into(),device.stream)},"stable binary mask sort")
}
pub fn mean(x:&Tensor,y:&Tensor,dims:&[i64],device:&Device)->Result<()>{
    reduction(x,y,dims,false,device)
}
pub fn norm2(x:&Tensor,y:&Tensor,dims:&[i64],device:&Device)->Result<()>{
    if x.dtype!="torch.float32"{return Err("norm2 dtype outside port".into());}
    reduction(x,y,dims,true,device)
}
fn reduction(x:&Tensor,y:&Tensor,dims:&[i64],norm:bool,device:&Device)->Result<()>{
    if x.dtype!=y.dtype || y.strides!=contiguous_strides(&y.shape){
        return Err("mean requires preserved dtype and contiguous output".into());
    }
    let rank=x.shape.len() as i64;let mut mask=0u32;
    for &dim in dims{
        let dim=if dim<0{dim+rank}else{dim};
        if dim<0 || dim>=rank{return Err("mean axis outside input rank".into());}
        mask|=1<<dim;
    }
    if mask==0{mask=(1<<rank)-1;}
    let shape=x.shape.iter().map(|&v|v as i64).collect::<Vec<_>>();
    let strides=x.strides.iter().map(|&v|v as i64).collect::<Vec<_>>();
    let count=x.shape.iter().enumerate().filter(|(i,_)|mask&(1<<i)!=0).map(|(_,v)|*v).product::<usize>();
    if count==0 || y.elements()?.checked_mul(count)!=Some(x.elements()?){return Err("mean extent differs".into());}
    let code=if norm{unsafe{ltx25_norm2(x.ptr() as _,y.ptr() as _,rank.try_into()?,shape.as_ptr(),strides.as_ptr(),mask,device.stream)}}
        else{unsafe{ltx25_mean(x.ptr() as _,y.ptr() as _,dtype(&x.dtype)?.0,rank.try_into()?,shape.as_ptr(),strides.as_ptr(),mask,device.stream)}};
    check(code,"PyTorch mean/norm2 reduction")
}
pub fn copy(x:&Tensor,y:&Tensor,stream:Stream)->Result<()>{
    if x.elements()?!=y.elements()? {return Err("copy element counts differ".into());}
    native_core_infra::execution_start::enqueue_if_armed(stream)?;
    check(unsafe{ltx25_tensor_copy(x.view()?,y.view()?,y.elements()?.try_into()?,stream)},"strided copy")
}
pub fn fill(y:&Tensor,start:f32,step:f32,stream:Stream)->Result<()>{
    let n=y.elements()?;
    if !start.is_finite() || !step.is_finite() || (step!=0. &&
        (start.abs().max((start+(n.saturating_sub(1) as f32)*step).abs())>16777216.)){
        return Err("factory range outside ported exact integer extent".into());
    }
    native_core_infra::execution_start::enqueue_if_armed(stream)?;
    check(unsafe{ltx25_tensor_fill(y.view()?,n.try_into()?,start,step,stream)},"constant/range factory")
}
pub fn elementwise(op:i32,a:&Tensor,b:Option<&Tensor>,c:Option<&Tensor>,y:&Tensor,
    scalar:f32,alpha:f32,stream:Stream)->Result<()>{
    native_core_infra::execution_start::enqueue_if_armed(stream)?;
    check(unsafe{ltx25_tensor_elementwise(op,a.view()?,b.unwrap_or(a).view()?,c.unwrap_or(a).view()?,y.view()?,
        y.elements()?.try_into()?,scalar,alpha,b.is_none().into(),stream)},"eager elementwise")
}
/// One op of a fused elementwise run: `op` as `elementwise`'s code (50: dtype cast), operands
/// `a`/`b`/`c` as registers (inputs first, then one per earlier op), `round` its eager output dtype
/// (0 bf16, 1 fp32).
#[repr(C)]
#[derive(Clone,Copy,Debug)]
pub struct FuseOp {pub op:c_int,pub a:c_int,pub b:c_int,pub c:c_int,pub round:c_int,pub scalar:f32,pub alpha:f32,pub scalar_b:c_int}
/// Runs `ops` per element of the outputs' shape and stores register `out_regs[k]` into `outputs[k]`.
pub fn fused(inputs:&[&Tensor],outputs:&[&Tensor],out_regs:&[i32],ops:&[FuseOp],stream:Stream)->Result<()>{
    if outputs.is_empty() || out_regs.len()!=outputs.len(){return Err("fused elementwise outputs".into());}
    let views=inputs.iter().map(|t|t.view()).collect::<Result<Vec<_>>>()?;
    let targets=outputs.iter().map(|t|t.view()).collect::<Result<Vec<_>>>()?;
    native_core_infra::execution_start::enqueue_if_armed(stream)?;
    check(unsafe{ltx25_fused_elementwise(views.as_ptr(),views.len().try_into()?,targets.as_ptr(),out_regs.as_ptr(),
        targets.len().try_into()?,ops.as_ptr(),ops.len().try_into()?,outputs[0].elements()?.try_into()?,stream)},"fused elementwise")
}
pub fn rms(x:&Tensor,w:Option<&Tensor>,y:&Tensor,rstd:Option<&Tensor>,eps:f32,device:&Device)->Result<()>{
    if y.strides!=contiguous_strides(&y.shape) {
        return Err("RMS output must be contiguous".into());
    }
    let materialized=if x.strides!=contiguous_strides(&x.shape){Some(x.contiguous(device)?)}else{None};
    let x=materialized.as_ref().unwrap_or(x);
    let cols=*x.shape.last().ok_or("RMS requires a last dimension")?;
    check(unsafe{ltx25_tensor_rms(x.ptr() as _,w.map_or(0,Tensor::ptr) as _,y.ptr() as _,
        rstd.map_or(0,Tensor::ptr) as _,(x.elements()?/cols).try_into()?,cols.try_into()?,eps,dtype(&x.dtype)?.0,
        device.stream)},"torch RMS reduction")
}
pub fn layer_norm(x:&Tensor,w:Option<&Tensor>,bias:Option<&Tensor>,y:&Tensor,
    mean:Option<&Tensor>,rstd:Option<&Tensor>,eps:f32,device:&Device)->Result<()>{
    if x.strides!=contiguous_strides(&x.shape) || y.strides!=contiguous_strides(&y.shape){
        return Err("LayerNorm input/output must be contiguous".into());
    }
    if mean.into_iter().chain(rstd).any(|t|t.dtype!="torch.float32"){
        return Err("LayerNorm statistics must use FP32".into());
    }
    let cols=*x.shape.last().ok_or("LayerNorm needs a last dimension")?;
    check(unsafe{ltx25_tensor_layer_norm(x.ptr() as _,w.map_or(0,Tensor::ptr) as _,
        bias.map_or(0,Tensor::ptr) as _,y.ptr() as _,mean.map_or(0,Tensor::ptr) as _,
        rstd.map_or(0,Tensor::ptr) as _,(x.elements()?/cols).try_into()?,cols.try_into()?,eps,
        dtype(&x.dtype)?.0,device.stream)},"LayerNorm Welford reduction")
}
pub fn group_norm(x:&Tensor,w:Option<&Tensor>,bias:Option<&Tensor>,y:&Tensor,groups:usize,eps:f32,device:&Device)->Result<()>{
    if x.shape.len()<3 || x.strides!=contiguous_strides(&x.shape) || y.strides!=x.strides || x.dtype!=y.dtype{
        return Err("GroupNorm requires contiguous NCHW/NCDHW".into());
    }
    if w.into_iter().chain(bias).any(|v|v.dtype!=x.dtype || v.elements().ok()!=Some(x.shape[1])){
        return Err("mixed-type GroupNorm parameters have not been ported".into());
    }
    let stats=Buffer::new(x.shape[0]*groups*8)?;
    check(unsafe{ltx25_group_norm(x.ptr() as _,w.map_or(0,Tensor::ptr) as _,bias.map_or(0,Tensor::ptr) as _,
        y.ptr() as _,stats.ptr(),x.shape[0].try_into()?,x.shape[1].try_into()?,
        (x.elements()?/x.shape[0]/x.shape[1]).try_into()?,groups.try_into()?,eps,dtype(&x.dtype)?.0,device.stream)},"GroupNorm Welford reduction")
}
