//! The shared pinned cuDNN convolution backend, with Torch's input materialization.
use crate::{cuda::{Device,Stream},tensor::{self,Tensor},Result};
use std::ffi::{c_char,c_int,c_void,CStr};
#[repr(C)]
struct Spec {
    ndim:i32,dtype:i32,transposed:i32,allow_tf32:i32,deterministic:i32,channels_last:i32,
    input_dims:[i64;5],input_strides:[i64;5],weight_dims:[i64;5],weight_strides:[i64;5],
    output_dims:[i64;5],output_strides:[i64;5],padding:[i64;3],stride:[i64;3],dilation:[i64;3],
}
extern "C" {
    fn infra_conv_create(context:*mut *mut c_void,plan_limit:usize,workspace_limit:usize)->c_int;
    fn infra_conv_forward(context:*mut c_void,spec:*const Spec,input:*const c_void,weight:*const c_void,
        output:*mut c_void,stream:Stream)->c_int;
    fn infra_conv_error()->*const c_char;
    fn infra_conv_destroy(context:*mut c_void)->c_int;
}
fn error()->String{unsafe{CStr::from_ptr(infra_conv_error())}.to_string_lossy().into_owned()}
pub struct Conv {context:*mut c_void}
fn fixed(values:&[usize])->Result<[i64;5]>{
    if values.len()>5{return Err("convolution rank".into());}
    let mut out=[1;5];for (i,&n) in values.iter().enumerate(){out[i]=n.try_into()?;}Ok(out)
}
fn layout(x:&Tensor,last:bool,device:&Device)->Result<Tensor>{
    let mut strides=tensor::contiguous_strides(&x.shape);
    if last{
        let mut current=x.shape[1];strides[1]=1;
        for dim in (2..x.shape.len()).rev(){strides[dim]=current;current*=x.shape[dim];}
        strides[0]=current;
    }
    if x.strides==strides{return Ok(x.clone());}
    let out=Tensor::allocate(x.shape.clone(),strides,x.dtype.clone(),device)?;
    tensor::copy(x,&out,device.stream)?;Ok(out)
}
impl Conv{
    pub fn new()->Result<Self>{
        let mut context=std::ptr::null_mut();
        if unsafe{infra_conv_create(&mut context,512,32usize<<30)}!=0{return Err(error().into());}
        Ok(Self{context})
    }
    pub fn forward(&self,x:&Tensor,w:&Tensor,bias:Option<&Tensor>,y:&Tensor,stride:&[usize],padding:&[usize],
        dilation:&[usize],groups:usize,transposed:bool,device:&Device)->Result<()>{
        let rank=x.shape.len();
        if rank==3{
            let lift=|v:&Tensor|{let mut out=v.clone();out.shape.insert(2,1);out.strides.insert(2,v.shape[2]*v.strides[2]);out};
            if stride.len()!=1 || padding.len()!=1 || dilation.len()!=1{return Err("1D convolution spatial parameters".into());}
            return self.forward(&lift(x),&lift(w),bias,&lift(y),&[1,stride[0]],&[0,padding[0]],&[1,dilation[0]],groups,transposed,device);
        }
        if !matches!(rank,4|5) || w.shape.len()!=rank || y.shape.len()!=rank || x.dtype!=w.dtype || y.dtype!=x.dtype ||
            stride.len()!=rank-2 || padding.len()!=rank-2 || dilation.len()!=rank-2{
            return Err("convolution descriptor is outside ported rank/dtype".into());
        }
        if groups==0 || (!transposed && (x.shape[1]!=w.shape[1]*groups || y.shape[1]!=w.shape[0]))
            || (transposed && (x.shape[1]!=w.shape[0] || y.shape[1]!=w.shape[1]*groups)){
            return Err("convolution grouped channel dimensions differ".into());
        }
        let channels_last=y.strides[1]==1 && y.shape[1]>1;
        let x=layout(x,channels_last,device)?;let w=layout(w,channels_last,device)?;
        let mut spec=Spec{ndim:rank as i32,dtype:match x.dtype.as_str(){"torch.float32"=>0,"torch.bfloat16"=>2,
            _=>return Err("convolution dtype".into())},transposed:transposed.into(),allow_tf32:0,deterministic:1,channels_last:channels_last.into(),
            input_dims:fixed(&x.shape)?,input_strides:fixed(&x.strides)?,weight_dims:fixed(&w.shape)?,weight_strides:fixed(&w.strides)?,
            output_dims:fixed(&y.shape)?,output_strides:fixed(&y.strides)?,padding:[0;3],stride:[1;3],dilation:[1;3]};
        for i in 0..rank-2{spec.stride[i]=stride[i].try_into()?;spec.padding[i]=padding[i].try_into()?;spec.dilation[i]=dilation[i].try_into()?;}
        if unsafe{infra_conv_forward(self.context,&spec,x.ptr() as _,w.ptr() as _,y.ptr() as _,device.stream)}!=0{
            return Err(error().into());
        }
        if let Some(bias)=bias{
            if bias.dtype!=y.dtype || bias.elements()?!=y.shape[1]{return Err("convolution bias extent/dtype".into());}
            let mut broadcast=bias.clone();broadcast.shape=vec![1;rank];broadcast.shape[1]=y.shape[1];
            broadcast.strides=vec![0;rank];broadcast.strides[1]=1;
            tensor::elementwise(20,y,Some(&broadcast),None,y,0.,1.,device.stream)?;
        }
        Ok(())
    }
}
impl Drop for Conv{fn drop(&mut self){unsafe{infra_conv_destroy(self.context);}}}
