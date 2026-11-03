//! FL2VA window geometry of the original windowed reference planner.
//! This is request planning only. It does not enable long-video inference.
use crate::Result;
use serde::Serialize;
use serde_json::{json, Value};

const NATIVE_GROUPS: usize = 20;
const WINDOW_GROUPS: usize = 11;
const OVERLAP_GROUPS: usize = 3;
const MIN_GROUPS: usize = 7;
const DECODE_GROUPS: usize = 20;
const DECODE_CONTEXT: usize = 1;

fn audio_count(groups: usize) -> usize {
    // Preserve Python's operation order and ties-to-even, including the
    // 5/3 * (1+4+4+4+4) expression used for the audio origin below.
    ((17 * groups + 5) as f64 / 24.0 * 40.0).round_ties_even() as usize
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Window {
    pub index: usize,
    pub groups: [usize; 2],
    pub latent_frames: [usize; 2],
    pub pixel_frames: usize,
    pub audio_latents: [usize; 2],
    pub is_first: bool,
    pub is_last: bool,
    pub overlap_latent_frames: usize,
    pub overlap_audio_latents: usize,
}

impl Window {
    fn new(index: usize, start: usize, groups: usize, total: usize, previous: Option<&Self>) -> Self {
        let is_last = start + groups == total;
        let audio_length = audio_count(groups);
        let max_start = audio_count(total) - audio_length;
        let audio_start = if is_last { max_start } else {
            ((5.0_f64 / 3.0 * 17.0 * start as f64).round_ties_even() as usize).min(max_start)
        };
        let latent_start = 5 * start;
        Self {
            index, groups: [start, start + groups], latent_frames: [latent_start, 5 * (start + groups) + 2],
            pixel_frames: 17 * groups + 5, audio_latents: [audio_start, audio_start + audio_length],
            is_first: start == 0, is_last,
            overlap_latent_frames: previous.map_or(0, |p| p.latent_frames[1].saturating_sub(latent_start)),
            overlap_audio_latents: previous.map_or(0, |p| p.audio_latents[1].saturating_sub(audio_start)),
        }
    }

    pub fn anchors(&self, first: bool, last: bool, multiple: bool) -> Value {
        // The first portrait supplies carried anchors. With text-only input
        // there are no invented keyframes; a real last keyframe wins over it.
        json!({
            "first": if first {Some(if self.is_first {"real"} else {"carried"})} else {None},
            "last": if last && self.is_last {Some("real")}
                else if first && !self.is_last {Some("carried")}
                else if first && multiple {Some("portrait")} else {None},
        })
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct DecodeSlice {
    pub groups: [usize; 2],
    pub context_groups: [usize; 2],
    pub latents: [usize; 2],
    pub decoded_frames: usize,
    pub head: usize,
    pub tail: usize,
    pub frames: [usize; 2],
}

#[derive(Debug, Serialize)]
pub struct Plan {
    pub num_frames: usize,
    pub num_latent_frames: usize,
    pub num_audio_latents: usize,
    pub native_denoise: bool,
    pub sliced_decode: bool,
    pub windows: Vec<Window>,
    pub decode_slices: Vec<DecodeSlice>,
}

impl Plan {
    /// Input is already aligned by the serving contract. The duration ceiling
    /// is trusted deployment configuration, never a request-level override.
    pub fn new(num_frames: usize, max_duration_s: f64, force: bool) -> Result<Self> {
        if !max_duration_s.is_finite() || max_duration_s < 5.0 {
            return Err("long-video duration ceiling must be finite and at least 5 seconds".into());
        }
        if num_frames < 124 || num_frames % 17 != 5 || num_frames as f64 / 24.0 > max_duration_s {
            return Err("long-video frames must be on the 17n+5 grid and within the configured duration".into());
        }
        num_frames.checked_mul(2).ok_or("long-video audio geometry overflows")?;
        let total = (num_frames - 5) / 17;
        let mut windows = Vec::new();
        if total <= NATIVE_GROUPS {
            windows.push(Window::new(0, 0, total, total, None));
        } else {
            let mut start = 0;
            loop {
                let remaining = total - start;
                let groups = if remaining <= WINDOW_GROUPS {
                    if remaining < MIN_GROUPS { start = total - MIN_GROUPS; MIN_GROUPS } else { remaining }
                } else { WINDOW_GROUPS };
                windows.push(Window::new(windows.len(), start, groups, total, windows.last()));
                if start + groups == total { break; }
                start += WINDOW_GROUPS - OVERLAP_GROUPS;
            }
        }
        // The reference starts sliced decode at 24 groups, independently of the
        // denoiser's 20-group boundary. A short request retains one-shot decode.
        let sliced_decode = total >= 24;
        let slice_groups = if sliced_decode { DECODE_GROUPS } else { total };
        let mut decode_slices = Vec::new();
        let mut at = 0;
        for first in (0..total).step_by(slice_groups) {
            let last = (first + slice_groups).min(total);
            let lo = first.saturating_sub(DECODE_CONTEXT);
            let hi = (last + DECODE_CONTEXT).min(total);
            let head = 17 * (first - lo);
            let kept = 17 * (last - first) + if last == total { 5 } else { 0 };
            decode_slices.push(DecodeSlice {
                groups: [first, last], context_groups: [lo, hi], latents: [5 * lo, 5 * hi + 2],
                decoded_frames: 17 * (hi - lo) + 5, head, tail: head + kept, frames: [at, at + kept],
            });
            at += kept;
        }
        if at != num_frames { return Err("long-video decode coverage mismatch".into()); }
        Ok(Self { num_frames, num_latent_frames: 5 * total + 2, num_audio_latents: audio_count(total),
            native_denoise: total <= NATIVE_GROUPS && !force, sliced_decode, windows, decode_slices })
    }
}

/// Preserve the reference's prefix-then-suffix ramp construction. The denoiser
/// divides each accumulated velocity by its accumulated FP32 weight.
pub fn fuse_weights(count: usize, before: usize, after: usize) -> Result<Vec<f32>> {
    if count == 0 || before > count || after > count {
        return Err("invalid long-video overlap geometry".into());
    }
    let mut out = vec![1.0; count];
    for k in 0..before { out[k] = (k as f32 + 1.0) / (before as f32 + 1.0); }
    for k in 0..after { out[count - after + k] = (after as f32 - k as f32) / (after as f32 + 1.0); }
    Ok(out)
}

pub fn describe(request: &Value) -> Result<Value> {
    let frames = usize::try_from(request["num_frames"].as_u64().ok_or("num_frames must be an integer")?)?;
    let max_duration = request.get("max_duration_s").map_or(Ok(60.0), |v| v.as_f64().ok_or("invalid duration ceiling"))?;
    let force = request.get("force").map_or(Ok(false), |v| v.as_bool().ok_or("force must be a boolean"))?;
    let plan = Plan::new(frames, max_duration, force)?;
    let mut result = serde_json::to_value(&plan)?;
    for (i, window) in plan.windows.iter().enumerate() {
        let next = plan.windows.get(i + 1);
        let multiple = plan.windows.len() > 1;
        result["windows"][i]["video_weights"] = json!(fuse_weights(window.latent_frames[1] - window.latent_frames[0],
            window.overlap_latent_frames, next.map_or(0, |w| w.overlap_latent_frames))?);
        result["windows"][i]["audio_weights"] = json!(fuse_weights(window.audio_latents[1] - window.audio_latents[0],
            window.overlap_audio_latents, next.map_or(0, |w| w.overlap_audio_latents))?);
        result["windows"][i]["anchor_layouts"] = json!({"text_only":window.anchors(false,false,multiple),
            "first":window.anchors(true,false,multiple),"first_last":window.anchors(true,true,multiple)});
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_denoise_decode_thresholds() {
        let native = Plan::new(345, 60.0, false).unwrap();
        assert!(native.native_denoise && !native.sliced_decode);
        assert!(!Plan::new(345, 60.0, true).unwrap().native_denoise);
        let split = Plan::new(396, 60.0, false).unwrap();
        assert!(!split.native_denoise && !split.sliced_decode);
        assert!(Plan::new(413, 60.0, false).unwrap().sliced_decode);
    }
    #[test]
    fn contract_rejects_before_work() {
        for frames in [0, 123, 360, 1450, usize::MAX] { assert!(Plan::new(frames,60.0,false).is_err()); }
        assert!(Plan::new(1433,60.0,false).is_ok());
        for ceiling in [0.0,f64::NAN,f64::INFINITY] { assert!(Plan::new(124,ceiling,false).is_err()); }
    }
    #[test]
    fn pulled_tail_uses_normalized_accumulation() {
        let plan = Plan::new(17 * 21 + 5, 60.0, false).unwrap();
        assert_eq!(plan.windows.iter().map(|w| w.groups).collect::<Vec<_>>(),vec![[0,11],[8,19],[14,21]]);
        let mut coverage = vec![0.0;plan.num_latent_frames];
        for (i,w) in plan.windows.iter().enumerate() {
            for (j,weight) in fuse_weights(w.latent_frames[1]-w.latent_frames[0],w.overlap_latent_frames,
                plan.windows.get(i+1).map_or(0,|x|x.overlap_latent_frames)).unwrap().iter().enumerate() {
                coverage[w.latent_frames[0]+j] += weight;
            }
        }
        assert!(coverage.iter().all(|&v|v>0.0));
        assert_eq!(plan.decode_slices.last().unwrap().frames[1],362);
        assert_eq!(plan.windows.last().unwrap().audio_latents[1],plan.num_audio_latents);
    }
}
