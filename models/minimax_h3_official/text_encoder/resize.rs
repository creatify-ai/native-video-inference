//! Native port of pinned Transformers smart_resize and PyTorch uint8 bicubic-AA.
//! See reference_sources/ for upstream source identities and PyTorch license.
use crate::Result;

pub fn smart(width:usize,height:usize,factor:usize,min_pixels:usize,max_pixels:usize)->Result<(usize,usize)> {
    let area=width.checked_mul(height).ok_or("canvas area overflow")?;
    if width==0 || height==0 || factor==0 || min_pixels==0 || min_pixels>max_pixels ||
       (width.max(height) as f64)/(width.min(height) as f64)>200.0 {
        return Err("invalid Qwen canvas/aspect ratio or pixel limits".into());
    }
    let mut h=(height as f64/factor as f64).round_ties_even() as usize*factor;
    let mut w=(width as f64/factor as f64).round_ties_even() as usize*factor;
    let rounded=h.checked_mul(w).ok_or("resized area overflow")?;
    if rounded>max_pixels {
        let beta=(area as f64/max_pixels as f64).sqrt();
        h=factor.max((height as f64/beta/factor as f64).floor() as usize*factor);
        w=factor.max((width as f64/beta/factor as f64).floor() as usize*factor);
    }else if rounded<min_pixels {
        let beta=(min_pixels as f64/area as f64).sqrt();
        h=(height as f64*beta/factor as f64).ceil() as usize*factor;
        w=(width as f64*beta/factor as f64).ceil() as usize*factor;
    }
    if w==0 || h==0 || w.checked_mul(h).is_none(){return Err("invalid smart_resize output".into());}
    Ok((w,h))
}
fn filter(value:f64)->f64 {
    let x=value.abs();let a=-0.5;
    if x<1.0 {((a+2.0)*x-(a+3.0))*x*x+1.0}
    else if x<2.0 {((a*x-5.0*a)*x+8.0*a)*x-4.0*a}
    else {0.0}
}
struct Coeff {start:usize,weights:Vec<i16>}
fn coefficients(input:usize,output:usize)->(Vec<Coeff>,u32) {
    let scale=input as f64/output as f64;
    let support=if scale>=1.0{2.0*scale}else{2.0};
    let max_size=(support.ceil() as usize)*2+1;
    let invscale=if scale>=1.0{1.0/scale}else{1.0};
    let mut floats=Vec::with_capacity(output);let mut maximum=0.0f64;
    for i in 0..output {
        let center=scale*(i as f64+0.5);
        let start=((center-support+0.5) as i64).max(0) as usize;
        let end=((center+support+0.5) as usize).min(input);
        let size=(end-start).min(max_size);
        let mut weights=Vec::with_capacity(size);let mut total=0.0;
        for j in 0..size {let v=filter((j as f64+start as f64-center+0.5)*invscale);weights.push(v);total+=v;}
        if total!=0.0 {for v in &mut weights {*v/=total;maximum=maximum.max(*v);}}
        floats.push((start,weights));
    }
    let mut precision=0u32;
    while precision<22 {
        if (0.5+maximum*((1u64<<(precision+1)) as f64)) as i64>=1<<15{break;}
        precision+=1;
    }
    let values=floats.into_iter().map(|(start,ws)|Coeff{start,weights:ws.into_iter().map(|w|{
        let v=w*((1u64<<precision) as f64);(if v<0.0{-0.5+v}else{0.5+v}) as i16
    }).collect()}).collect();
    (values,precision)
}
pub fn rgb(input:&[u8],width:usize,height:usize,out_width:usize,out_height:usize)->Result<Vec<u8>> {
    let size=width.checked_mul(height).and_then(|n|n.checked_mul(3)).ok_or("RGB byte overflow")?;
    let out_size=out_width.checked_mul(out_height).and_then(|n|n.checked_mul(3)).ok_or("resize byte overflow")?;
    if width==0||height==0||out_width==0||out_height==0||input.len()!=size{return Err("invalid RGB resize dimensions".into());}
    if (width,height)==(out_width,out_height){return Ok(input.to_vec());}
    let horizontal=if width==out_width {input.to_vec()}else{
        let (coeff,precision)=coefficients(width,out_width);let mut out=vec![0u8;height*out_width*3];
        for y in 0..height {for (x,c) in coeff.iter().enumerate(){for channel in 0..3 {
            let mut acc=1i32<<(precision-1);
            for (j,&weight) in c.weights.iter().enumerate(){acc+=input[(y*width+c.start+j)*3+channel] as i32*weight as i32;}
            out[(y*out_width+x)*3+channel]=(acc>>precision).clamp(0,255) as u8;
        }}}
        out
    };
    if height==out_height{return Ok(horizontal);}
    let (coeff,precision)=coefficients(height,out_height);let mut out=vec![0u8;out_size];
    for (y,c) in coeff.iter().enumerate(){for x in 0..out_width*3 {
        let mut acc=1i32<<(precision-1);
        for (j,&weight) in c.weights.iter().enumerate(){acc+=horizontal[(c.start+j)*out_width*3+x] as i32*weight as i32;}
        out[y*out_width*3+x]=(acc>>precision).clamp(0,255) as u8;
    }}
    Ok(out)
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn smart_identity_and_resize(){for (w,h) in [(768,768),(768,1344),(1344,768),(1440,1440)]{assert_eq!(smart(w,h,32,65536,16777216).unwrap(),(w,h));}assert_eq!(smart(769,1345,32,65536,16777216).unwrap(),(768,1344));assert_eq!(smart(4097,4097,32,65536,16777216).unwrap(),(4096,4096));assert_eq!(smart(4160,4160,32,65536,16777216).unwrap(),(4096,4096));assert_eq!(smart(128,128,32,65536,16777216).unwrap(),(256,256));}
    #[test] fn python_ties_even(){assert_eq!(smart(784,816,32,1,16777216).unwrap(),(768,832));}
    #[test] fn invalid_geometry(){for (w,h) in [(0,1),(1,0),(1,201),(usize::MAX,3)]{assert!(smart(w,h,32,65536,16777216).is_err());}}
    #[test] fn preserve_constant(){for (w,h,ow,oh) in [(7,11,32,16),(51,23,9,3),(3,3,3,7)]{assert_eq!(rgb(&vec![123;w*h*3],w,h,ow,oh).unwrap(),vec![123;ow*oh*3]);}}
}
