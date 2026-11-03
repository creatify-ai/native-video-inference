use crate::{cuda::{Device,Stream,check,upload_into},gold::sha,tensor::{self,Tensor},Result};
use serde_json::{json,Value};
use std::{ffi::{c_char,c_int,c_void,CStr},path::Path};
extern "C" {
    fn ltx25_media_decode(source:*const u8,length:usize,crf:c_int,decoded:*mut *mut u8,preprocessed:*mut *mut u8,width:*mut c_int,height:*mut c_int)->c_int;
    fn ltx25_media_free(pointer:*mut c_void);
    fn ltx25_media_error()->*const c_char;
    fn ltx25_bilinear_rgb(source:*const u8,out:*mut f32,ih:c_int,iw:c_int,oh:c_int,ow:c_int,stream:Stream)->c_int;
    fn ltx25_encode_mp4(path:*const c_char,rgb:*const f32,frames:usize,width:c_int,height:c_int,fps:c_int,
        audio:*const f32,samples:usize,rate:c_int)->c_int;
    fn ltx25_encode_error()->*const c_char;
}
pub struct Image {pub rgb:Vec<u8>,pub width:usize,pub height:usize,pub decoded_sha:String}
pub fn decode(path:&Path,crf:i32)->Result<Image>{
    let raw=std::fs::read(path)?;let mut decoded=std::ptr::null_mut();let mut processed=std::ptr::null_mut();let mut width=0;let mut height=0;
    let status=unsafe{ltx25_media_decode(raw.as_ptr(),raw.len(),crf,&mut decoded,&mut processed,&mut width,&mut height)};
    if status!=0{return Err(unsafe{CStr::from_ptr(ltx25_media_error())}.to_string_lossy().into_owned().into());}
    let size=usize::try_from(width)?.checked_mul(usize::try_from(height)?).and_then(|n|n.checked_mul(3)).ok_or("image extent overflow")?;
    let decoded_sha=sha(unsafe{std::slice::from_raw_parts(decoded,size)});
    let rgb=unsafe{std::slice::from_raw_parts(processed,size)}.to_vec();
    unsafe{ltx25_media_free(decoded as _);ltx25_media_free(processed as _);}
    Ok(Image{rgb,width:width as usize,height:height as usize,decoded_sha})
}
pub fn prepare(image:&Image,height:usize,width:usize,device:&Device)->Result<Tensor>{
    let scale=(height as f64/image.height as f64).max(width as f64/image.width as f64);
    let oh=(image.height as f64*scale).ceil() as usize;let ow=(image.width as f64*scale).ceil() as usize;
    let source=Tensor::contiguous_shape(vec![image.height,image.width,3],"torch.uint8".into(),device)?;
    upload_into(&source.storage,&image.rgb,device)?;
    let output=Tensor::allocate(vec![1,3,oh,ow],vec![oh*ow*3,1,ow*3,3],"torch.float32".into(),device)?;
    check(unsafe{ltx25_bilinear_rgb(source.ptr() as _,output.ptr() as _,image.height.try_into()?,image.width.try_into()?,
        oh.try_into()?,ow.try_into()?,device.stream)},"PyTorch bilinear RGB")?;
    let mut crop=output;crop.offset=(((oh-height)/2*ow+(ow-width)/2)*3)*4;
    crop.shape=vec![1,3,1,height,width];crop.strides=vec![oh*ow*3,1,oh*ow*3,ow*3,3];
    let divided=Tensor::contiguous_shape(crop.shape.clone(),"torch.float32".into(),device)?;
    tensor::elementwise(23,&crop,None,None,&divided,127.5,1.,device.stream)?;
    let normalized=Tensor::contiguous_shape(crop.shape.clone(),"torch.float32".into(),device)?;
    tensor::elementwise(22,&divided,None,None,&normalized,1.,1.,device.stream)?;
    let result=Tensor::contiguous_shape(crop.shape,"torch.bfloat16".into(),device)?;
    tensor::copy(&normalized,&result,device.stream)?;Ok(result)
}
pub fn encode(path:&Path,frames:&[Tensor],audio:&Tensor,fps:usize,rate:usize,device:&Device)->Result<Value>{
    let first=frames.first().ok_or("decoded video is empty")?;
    if first.shape.len()!=4 || first.shape[3]!=3 || audio.shape.len()!=2 || audio.shape[0]!=2{return Err("delivery tensor shape".into());}
    let height=first.shape[1];let width=first.shape[2];let mut rgb=Vec::new();let mut count=0;
    for chunk in frames{
        if chunk.shape.len()!=4 || chunk.shape[1..]!=first.shape[1..]{return Err("delivery chunk dimensions differ".into());}
        let value=Tensor::contiguous_shape(chunk.shape.clone(),"torch.float32".into(),device)?;
        tensor::copy(chunk,&value,device.stream)?;
        rgb.extend(value.download(device)?.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())));
        count+=chunk.shape[0];
    }
    let sound=audio.download(device)?.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    encode_host(path,&rgb,count,width,height,fps,&sound,audio.shape[1],rate)
}
pub fn encode_host(path:&Path,rgb:&[f32],count:usize,width:usize,height:usize,fps:usize,
    sound:&[f32],samples:usize,rate:usize)->Result<Value>{
    if count==0 || width==0 || height==0 || fps==0 || rate==0 || samples==0
        || width%2!=0 || height%2!=0 || count.checked_mul(width).and_then(|n|n.checked_mul(height)).and_then(|n|n.checked_mul(3))!=Some(rgb.len())
        || samples.checked_mul(2)!=Some(sound.len()){return Err("delivery host shape differs".into());}
    let filename=std::ffi::CString::new(path.to_str().ok_or("delivery filename")?)?;
    let status=unsafe{ltx25_encode_mp4(filename.as_ptr(),rgb.as_ptr(),count,width.try_into()?,height.try_into()?,fps.try_into()?,
        sound.as_ptr(),samples,rate.try_into()?)};
    if status!=0{return Err(unsafe{CStr::from_ptr(ltx25_encode_error())}.to_string_lossy().into_owned().into());}
    let file=std::fs::read(path)?;
    Ok(json!({"path":path,"frames":count,"height":height,"width":width,"fps":fps,"sampling_rate":rate,
        "bytes":file.len(),"sha256":sha(&file),"exact_mp4_required":false}))
}
