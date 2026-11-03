//! Float WAV transport for the native decoder's channel-major stereo output.
use crate::Result;

pub fn wav_bytes(raw:&[u8],channels:u16,rate:u32)->Result<Vec<u8>>{
    if !(1..=2).contains(&channels)||rate==0||raw.is_empty()||raw.len()%(channels as usize*4)!=0{
        return Err("audio-wav requires nonempty planar FP32 mono/stereo audio".into());
    }
    let data_bytes=u32::try_from(raw.len())?;let samples=raw.len()/(channels as usize*4);
    let mut out=Vec::with_capacity(44+raw.len());
    out.extend_from_slice(b"RIFF");out.extend_from_slice(&data_bytes.checked_add(36).ok_or("WAV too large")?.to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes()); // WAVE_FORMAT_IEEE_FLOAT
    out.extend_from_slice(&channels.to_le_bytes());out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&rate.checked_mul(channels as u32*4).ok_or("WAV rate overflow")?.to_le_bytes());
    out.extend_from_slice(&(channels*4).to_le_bytes());out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"data");out.extend_from_slice(&data_bytes.to_le_bytes());
    for sample in 0..samples{for channel in 0..channels as usize{
        let i=(channel*samples+sample)*4;let value=f32::from_le_bytes(raw[i..i+4].try_into().unwrap());
        if !value.is_finite(){return Err("nonfinite audio sample".into());}
        out.extend_from_slice(&value.clamp(-1.,1.).to_le_bytes());
    }}
    Ok(out)
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn wav_interleaves_clips_and_preserves_float_bits(){
        let values=[-2f32,-0.,0.5,2.,0.25,-0.25];
        let raw=values.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
        let wav=wav_bytes(&raw,2,32000).unwrap();
        assert_eq!(&wav[..4],b"RIFF");assert_eq!(&wav[8..16],b"WAVEfmt ");
        assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()),3);
        let expected=[-1f32,1.,-0.,0.25,0.5,-0.25].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
        assert_eq!(&wav[44..],expected);assert_eq!(wav.len(),44+raw.len());
    }
    #[test]
    fn wav_rejects_incomplete_or_nonfinite_audio(){
        for raw in [vec![],vec![0;7],f32::NAN.to_le_bytes().to_vec()]{assert!(wav_bytes(&raw,2,32000).is_err());}
        let nonfinite=[f32::NAN,0.].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
        assert!(wav_bytes(&nonfinite,2,32000).is_err());
        assert!(wav_bytes(&[0;8],3,32000).is_err());assert!(wav_bytes(&[0;8],2,0).is_err());
    }
}
