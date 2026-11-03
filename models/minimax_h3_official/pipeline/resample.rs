//! Torchaudio transforms.Resample default sinc/Hann arithmetic, without Torch.
//! Original: torchaudio.functional.functional (_get/_apply_sinc_resample_kernel).
//! Truncate at source rate, duplicate mono, then one FP32 FMA convolution.
use crate::Result;

pub fn normalize(input:&[f32],channels:usize,rate:usize,frames:usize)->Result<Vec<f32>> {
    normalize_to_rate(input,channels,rate,frames,32000)
}

/// The delivery copy is resampled directly from the upload, never from 32 kHz.
pub fn normalize_to_rate(input:&[f32],channels:usize,rate:usize,frames:usize,target:usize)->Result<Vec<f32>> {
    if ![32000,48000].contains(&target){return Err("unsupported native audio target rate".into());}
    if ![1,2].contains(&channels) || rate==0 || rate>192000 || input.is_empty() || input.len()%channels!=0 || !(crate::MIN_RENDER_FRAMES..=1433).contains(&frames) {
        return Err("audio normalization requires mono/stereo, 1..192000 Hz and an admitted duration".into());
    }
    if input.iter().any(|x|!x.is_finite()){return Err("nonfinite decoded audio".into());}
    let source_len=input.len()/channels;
    let samples=source_len.min((frames as f64/24.0*rate as f64) as usize);
    if samples==0{return Err("empty reference soundtrack".into());}
    if rate==target {
        return Ok((0..2).flat_map(|c|input[(c%channels)*source_len..(c%channels)*source_len+samples].iter().copied()).collect());
    }
    let (mut a,mut b)=(rate,target);while b!=0{(a,b)=(b,a%b);}
    let (orig,next)=(rate/a,target/a);
    let base=orig.min(next) as f64*0.99;
    let width=(6.0*orig as f64/base).ceil() as usize;
    let taps=2*width+orig;
    // Bound pathological coprime source rates before allocating the polyphase kernel.
    if next.checked_mul(taps).is_none_or(|n|n>16_777_216){return Err("audio resampling kernel exceeds memory budget".into());}
    let mut kernel=Vec::with_capacity(next*taps);
    for phase in 0..next {
        // The original default constructs the phase offset in FP32 and the
        // index grid in FP64; making both FP64 changes 44.1 kHz coefficients.
        let offset=-(phase as f32)/(next as f32);
        for tap in 0..taps {
            let t=((offset as f64+(tap as f64-width as f64)/orig as f64)*base).clamp(-6.0,6.0);
            let window=(t*std::f64::consts::PI/6.0/2.0).cos();
            let t=t*std::f64::consts::PI;
            let sinc=if t==0.0{1.0}else{t.sin()/t};
            kernel.push((sinc*(window*window*(base/orig as f64))) as f32);
        }
    }
    // Original target length passes through a default FP32 scalar before ceil.
    let count=((next as f64*samples as f64/orig as f64) as f32).ceil() as usize;
    let mut output=vec![0f32;2*count];
    for channel in 0..2 {for out in 0..count {
        let start=(out/next*orig) as isize-width as isize;
        let weights=&kernel[(out%next)*taps..(out%next+1)*taps];
        let mut sum=0f32;
        for (tap,&weight) in weights.iter().enumerate(){
            let index=start+tap as isize;
            let x=if index>=0 && (index as usize)<samples{input[(channel%channels)*source_len+index as usize]}else{0.0};
            sum=x.mul_add(weight,sum);
        }
        output[channel*count+out]=sum;
    }}
    Ok(output)
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn native_rate_preserves_channels_and_signed_zero(){
        let x=[-0.0,0.25,-0.5];let y=normalize(&x,1,32000,124).unwrap();
        assert_eq!(y.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),x.iter().chain(&x).map(|x|x.to_bits()).collect::<Vec<_>>());
        assert_eq!(normalize(&[1.,2.,3.,4.],2,32000,124).unwrap(),vec![1.,2.,3.,4.]);
    }
    #[test] fn truncate_before_resampling(){
        let x=vec![0.25;32000*6];assert_eq!(normalize(&x,1,32000,124).unwrap().len(),2*(124.0/24.0*32000.0) as usize);
    }
    #[test] fn short_reference_soundtrack_is_truncated_before_resampling(){
        let x=vec![0.25;48000*6];
        for frames in [5,22,39,73,107,124]{
            let y=normalize(&x,1,48000,frames).unwrap();
            let source=normalize(&x[..(frames as f64/24.0*48000.0) as usize],1,48000,frames).unwrap();
            assert_eq!(y,source);
            assert_eq!(y.len(),2*(frames*32000).div_ceil(24));
        }
    }
    #[test] fn rejects_invalid_audio(){
        assert!(normalize(&[],1,32000,124).is_err());assert!(normalize(&[f32::NAN],1,32000,124).is_err());
        assert!(normalize(&[1.;6],3,48000,124).is_err());assert!(normalize(&[1.],1,0,124).is_err());
    }
}
