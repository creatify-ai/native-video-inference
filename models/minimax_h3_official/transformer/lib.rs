#[path="../support/startup_trace.rs"]
pub mod startup_trace;
#[path="../support/debug_audit.rs"]
pub mod debug_audit;
pub mod integrity;
pub mod pointwise_opt;
pub mod bsa_pack;
pub mod subblock;
pub mod timestep_plan;
pub mod execution;
pub mod request_transaction;
pub mod weight_shards;
pub mod model_payload;
pub mod scratch_plan;
pub mod attention_heads;
pub mod allocation_policy;
#[path="../support/limits.rs"]
pub mod limits;
#[path="../support/weight_loader.rs"]
pub mod weight_loader;
#[cfg(feature="cuda")]
pub mod host_cache;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub mod profile;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub use native_core_infra::tensor::{TensorSpec,f32_values,i64_values,bytes_i64};

/// Original MiniMaxH3BuildRowTimesteps: FP32 rows, sorted unique and inverse.
/// Qwen vision rows are still text embeddings here and inherit video_t.
pub fn row_timestep_indices(sequence:usize,video:&[i64],audio:&[i64],
    condition_video:usize,condition_audio:usize,video_t:f32,audio_t:f32,
    keyframe_noise_aug:f32)->Result<(Vec<f32>,Vec<i64>)> {
    if sequence==0 || condition_video>video.len() || condition_audio>audio.len() ||
        [video_t,audio_t,keyframe_noise_aug].iter().any(|x|!x.is_finite() || !(0.0..=1.0).contains(x)) {
        return Err("invalid row timestep schedule or condition counts".into());
    }
    let mut rows=vec![video_t;sequence];
    for &index in &video[..condition_video] {
        let row=usize::try_from(index).map_err(|_|"negative video row")?;
        *rows.get_mut(row).ok_or("video timestep row out of range")?=video_t.max(keyframe_noise_aug);
    }
    for (position,&index) in audio.iter().enumerate() {
        let row=usize::try_from(index).map_err(|_|"negative audio row")?;
        *rows.get_mut(row).ok_or("audio timestep row out of range")?=if position<condition_audio{1.0}else{audio_t};
    }
    let mut unique=rows.clone();unique.sort_by(|a,b|a.partial_cmp(b).unwrap());
    unique.dedup_by(|a,b|a.to_bits()==b.to_bits());
    let indices=rows.into_iter().map(|x|unique.iter().position(|y|y.to_bits()==x.to_bits()).unwrap() as i64).collect();
    Ok((unique,indices))
}

/// Select resident schedule rows by the exact FP32 timestep, not by the compact
/// request's unique-row ordinal. T2V omits the keyframe augmentation row.
pub fn resident_timestep_indices(unique:&[f32],indices:&[i64],resident:&[f32])->Result<Vec<i64>> {
    if unique.is_empty() || resident.is_empty() || unique.iter().chain(resident).any(|x|!x.is_finite()) ||
        resident.iter().enumerate().any(|(i,x)|resident[..i].iter().any(|v|v.to_bits()==x.to_bits())) {
        return Err("invalid or duplicate resident timestep values".into());
    }
    let map=unique.iter().map(|x|resident.iter().position(|v|v.to_bits()==x.to_bits())
        .map(|i|i as i64).ok_or("request timestep has no resident modulation table"))
        .collect::<std::result::Result<Vec<_>,_>>()?;
    indices.iter().map(|&index|usize::try_from(index).ok().and_then(|i|map.get(i)).copied()
        .ok_or_else(||"request timestep index out of range".into())).collect()
}

pub fn request_video_rows(width:usize,height:usize,frames:usize,keyframes:usize)->Result<(usize,usize)> {
    if keyframes>2 || width<32 || height<32 || width%32!=0 || height%32!=0 ||
        !(5..=345).contains(&frames) || (frames-5)%17!=0 {return Err("unsupported native video geometry/keyframes".into());}
    let rows=(width/32).checked_mul(height/32).ok_or("video row overflow")?;
    let latent_frames=(frames-5)/17*5+2;
    Ok(((latent_frames+keyframes).checked_mul(rows).ok_or("video row overflow")?,
        keyframes.checked_mul(rows).ok_or("conditioning row overflow")?))
}

pub fn finite_f32(data:&[u8])->Result<Value> {
    let _profile=crate::profile::HostSpan::new("output.finite_sha",data.len());
    let values=f32_values(data)?;
    if values.is_empty(){return Err("empty generated tensor".into());}
    let nonfinite=values.iter().filter(|x|!x.is_finite()).count();
    Ok(json!({"passed":nonfinite==0,"elements":values.len(),"nonfinite":nonfinite,
        "validation":"finite_only","reference_compared":false,
        "sha256":format!("{:x}",crate::startup_trace::measure("sha256","",(data).len(),||Sha256::digest(data)))}))
}

/// IEEE CRC-32, for optional intermediate localization only (not acceptance).
/// Matches Python zlib.crc32; numerical gates still compare every output bit.
pub fn diagnostic_crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut crc = i as u32;
        for _ in 0..8 { crc = (crc >> 1) ^ (0xedb88320 & 0u32.wrapping_sub(crc & 1)); }
        *entry = crc;
    }
    let mut crc = !0u32;
    for &byte in data { crc = table[((crc as u8) ^ byte) as usize] ^ (crc >> 8); }
    !crc
}

pub fn validate_packed_indices(tags: &[i64], video: &[i64], audio: &[i64], text: &[i64]) -> Result<()> {
    let mut covered = vec![false; tags.len()];
    for (name, indices) in [("video", video), ("audio", audio), ("text", text)] {
        for &index in indices {
            if index < 0 || index as usize >= tags.len() || covered[index as usize] {
                return Err(format!("{name} indices overlap or leave the packed sequence: {index}").into());
            }
            let tag = tags[index as usize];
            // Qwen's keyframe vision rows belong to the text embedding/index
            // group but carry video tag 0 for AdaLN (pinned FL2VA encoder).
            let allowed = match name { "video" => tag == 0, "audio" => tag == 2,
                "text" => tag == 0 || tag == 1, _ => unreachable!() };
            if !allowed { return Err(format!("{name} index {index} has invalid AdaLN tag {tag}").into()); }
            covered[index as usize] = true;
        }
    }
    if covered.iter().any(|&value| !value) { return Err("packed sequence has holes".into()); }
    Ok(())
}

pub fn compare(got: &[u8], want: &[u8], width: usize) -> Result<Value> {
    if got.len() != want.len() || got.is_empty() || ![2, 4].contains(&width) || got.len() % width != 0 {
        return Err("comparison size/dtype mismatch".into());
    }
    let (mut mismatches, mut nonfinite, mut max_abs, mut sum_abs, mut first, mut max_ulp) =
        (0usize, 0usize, 0f64, 0f64, None, 0u32);
    for (i, (a, b)) in got.chunks_exact(width).zip(want.chunks_exact(width)).enumerate() {
        let bits = |v: &[u8]| if width == 2 { (u16::from_le_bytes(v.try_into().unwrap()) as u32) << 16 }
            else { u32::from_le_bytes(v.try_into().unwrap()) };
        let (ab, bb) = (bits(a), bits(b));
        let (af, bf) = (f32::from_bits(ab) as f64, f32::from_bits(bb) as f64);
        if ab != bb { mismatches += 1; if first.is_none() { first = Some(i); } }
        if !af.is_finite() || !bf.is_finite() { nonfinite += 1; continue; }
        let delta = (af - bf).abs(); max_abs = max_abs.max(delta); sum_abs += delta;
        let ordered = |x: u32| if x & 0x80000000 == 0 { x | 0x80000000 } else { !x };
        let ulp = ordered(ab).abs_diff(ordered(bb));
        max_ulp = max_ulp.max(if width == 2 { ulp / 65536 } else { ulp });
    }
    Ok(json!({"passed": mismatches == 0 && nonfinite == 0, "elements":got.len()/width,
        "mismatches":mismatches, "nonfinite":nonfinite, "first_mismatch":first,
        "max_abs":max_abs, "mean_abs":sum_abs / (got.len()/width) as f64, "max_ulp":max_ulp}))
}

#[cfg(feature = "cuda")]
pub mod native;

#[cfg(test)]
mod tests {
    #[test]
    fn zero_one_two_keyframes_select_exact_resident_schedule_rows() {
        for count in 0..=2 {
            let (unique,indices)=super::row_timestep_indices(6,&[0,1,2],&[3,4],count,0,0.4,0.2,0.999).unwrap();
            let mapped=super::resident_timestep_indices(&unique,&indices,&[0.2,0.4,0.999]).unwrap();
            for (row,&index) in mapped.iter().enumerate() {
                let expected=if row<count {2}else if row==3||row==4 {0}else{1};
                assert_eq!(index,expected);
            }
        }
        assert_eq!(super::resident_timestep_indices(&[1.0],&[0,0],&[1.0]).unwrap(),vec![0,0]);
        assert!(super::resident_timestep_indices(&[0.3],&[0],&[0.2,0.4,0.999]).is_err());
        assert!(super::resident_timestep_indices(&[0.4],&[1],&[0.4]).is_err());
        assert!(super::resident_timestep_indices(&[0.4],&[0],&[0.4,0.4]).is_err());
        assert!(super::resident_timestep_indices(&[f32::NAN],&[0],&[0.4]).is_err());
    }
    #[test]
    fn geometry_and_condition_rows_follow_keyframe_count() {
        for (w,h,f) in [(768,768,345),(768,1376,124),(1376,768,192),(768,1376,5),(768,1376,22),(768,1376,39),(768,1376,107)] {
            for keyframes in 0..=2 {
                let rows=(w/32)*(h/32);
                assert_eq!(super::request_video_rows(w,h,f,keyframes).unwrap(),
                    (((f-5)/17*5+2+keyframes)*rows,keyframes*rows));
            }
        }
        assert!(super::request_video_rows(768,768,345,3).is_err());
        assert!(super::request_video_rows(768,768,346,0).is_err());
        assert!(super::request_video_rows(769,768,345,1).is_err());
        for frames in [0,1,4,6,24,108] {
            assert!(super::request_video_rows(768,1376,frames,1).is_err());
        }
    }
    use super::*;
    #[test]
    fn row_schedule_matches_original_sorted_unique_and_pinned_rows() {
        let (values,indices)=row_timestep_indices(8,&[3,4,7],&[5,6],2,0,0.4,0.2,0.999).unwrap();
        assert_eq!(values,vec![0.2,0.4,0.999]);
        assert_eq!(indices,vec![1,1,1,2,2,0,0,1]);
        let (values,indices)=row_timestep_indices(8,&[3,4,7],&[5,6],2,0,1.0,1.0,0.999).unwrap();
        assert_eq!(values,vec![1.0]);assert_eq!(indices,vec![0;8]);
        assert!(row_timestep_indices(2,&[-1],&[0],1,0,0.1,0.2,0.999).is_err());
    }
    #[test]
    fn generated_finiteness_is_not_a_reference_comparison() {
        let bytes=1.0f32.to_le_bytes();let result=finite_f32(&bytes).unwrap();
        assert_eq!(result["passed"],true);assert_eq!(result["reference_compared"],false);
        assert_eq!(finite_f32(&f32::NAN.to_le_bytes()).unwrap()["passed"],false);
    }
    #[test]
    fn diagnostic_crc_matches_ieee_vectors() {
        assert_eq!(diagnostic_crc32(b""), 0);
        assert_eq!(diagnostic_crc32(b"123456789"), 0xcbf43926);
    }
    #[test]
    fn exact_signed_zero_and_nonfinite() {
        assert!(compare(&[0, 0], &[0, 0], 2).unwrap()["passed"].as_bool().unwrap());
        assert!(!compare(&[0, 128], &[0, 0], 2).unwrap()["passed"].as_bool().unwrap());
        assert_eq!(compare(&[128, 127], &[128, 127], 2).unwrap()["nonfinite"], 1);
    }
    #[test]
    fn reject_original_stride_on_contiguous_storage() {
        let x = json!({"file":"x.raw","dtype":"torch.bfloat16","shape":[3,4],"stride":[24,1],
            "bytes":24,"contiguous":true,"storage_offset":0});
        assert!(TensorSpec::parse(&x).is_err());
    }
    #[test]
    fn reject_path_escape() {
        let x = json!({"file":"../x.raw","dtype":"torch.float32","shape":[1],"stride":[1],
            "bytes":4,"contiguous":true,"storage_offset":0});
        assert!(TensorSpec::parse(&x).is_err());
    }
    #[test]
    fn keyframe_vision_rows_use_video_adaln_in_text_embeddings() {
        let tags = [1, 0, 1, 0, 2, 0];
        assert!(validate_packed_indices(&tags, &[3, 5], &[4], &[0, 1, 2]).is_ok());
        assert!(validate_packed_indices(&tags, &[3, 5], &[4], &[0, 1]).is_err());
        assert!(validate_packed_indices(&tags, &[3, 5], &[4], &[0, 1, 1, 2]).is_err());
        assert!(validate_packed_indices(&tags, &[0, 5], &[4], &[1, 2, 3]).is_err());
        assert!(validate_packed_indices(&tags, &[3, 5], &[1], &[0, 2, 4]).is_err());
    }
}

#[cfg(feature="cuda")]
pub mod dynamic_router;

pub mod recipe;
pub mod numerics;

#[cfg(feature="cuda")]
pub mod server;
