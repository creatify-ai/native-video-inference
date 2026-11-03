//! Independent modality streams and dependent visual-condition noise in pinned SGLang.
use crate::{CpuGenerator, Result};

pub fn audio(seed:u64,count:usize)->Result<Vec<f32>> {
    CpuGenerator::new(seed).normal(count)
}

/// Each condition restarts the request seed, draws the complete dependent
/// temporal extent, and slices its own channel-major temporal prefix.
pub fn visual_condition(seed:u64,target_frames:usize,conditions:usize,frames:usize,
                        height:usize,width:usize)->Result<(Vec<f32>,usize)> {
    let full_frames=target_frames.checked_add(conditions).ok_or("condition noise extent overflow")?;
    if conditions==0 || frames==0 || frames>full_frames || height==0 || width==0 {
        return Err("invalid public visual-condition noise geometry".into());
    }
    let spatial=height.checked_mul(width).ok_or("condition noise shape overflow")?;
    let extent=full_frames.checked_mul(spatial).ok_or("condition noise shape overflow")?;
    let count=24usize.checked_mul(extent).ok_or("condition noise size overflow")?;
    let values=CpuGenerator::new(seed).normal(count)?;
    let mut prefix=Vec::with_capacity(24*frames*spatial);
    for channel in 0..24 {prefix.extend_from_slice(&values[channel*extent..channel*extent+frames*spatial]);}
    Ok((prefix,count))
}

#[cfg(all(test,feature="avx-noise",target_arch="x86_64"))]
mod tests {
    use super::*;
    #[test]
    fn audio_seed_zero_matches_the_independent_sglang_capture() {
        // Actual initial_audio_rows from pinned SGLang ad7a9b5, seed 0,
        // 124 frames at 24 fps: [2 * 207, 32] CPU FP32.
        assert_eq!(crate::sha256(&crate::f32_bytes(&audio(0,2*207*32).unwrap())),
            "70c5e8449b40e335e30fe31dbec2508324dc83d7fb21445878e2f2c28460ea49");
    }
}
