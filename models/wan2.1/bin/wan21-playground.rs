//! Parity rungs for the native Wan 2.1 port against golds recorded from the official PyTorch
//! pipeline on B200 (gold.rs describes the format; docs/porting-a-model.md the ladder). Each
//! subcommand prints a JSON report of per-tensor SHA-256 comparisons.
//!
//!   wan21-playground forward <gold run> <weights dir>   # first DiT forward, every block-0 op (fixture gold)
//!   wan21-playground sched <gold run>                   # schedule, CFG and every UniPC step from gold inputs
//!   wan21-playground encoders <gold run> <weights dir>  # T5 (prompt, negative prompt) and CLIP from gold inputs
//!   wan21-playground vae <gold run> <weights dir>       # VAE encode (conditioning) and decode from gold inputs
//!   wan21-playground frontend <gold run> <weights dir>  # PNG, size math, bicubic VAE input, tokens from inputs/
//!   wan21-playground e2e <gold run> <weights dir>       # the whole request from inputs/, every recorded tensor
//!   wan21-playground profile <gold run> <weights dir>   # DiT forwards on the first forward's inputs (Nsight range)
//!   wan21-playground load <prepared dir> <export dir>    # streaming in-place load, every tensor vs the export
//!   wan21-playground coldstart <inputs dir> <prepared dir> # wall-clock phases up to the first DiT forward
//!   wan21-playground ramcopy <prepared dir> <repeats>    # shared pinned-RAM upload, original stored bytes
//!
//! A gold run's inputs are its sibling `inputs/` (request.json, image.png). The weights dir is either
//! a converted export (dit/t5/clip/vae safetensors as the reference holds them, tokenizer.json) or a
//! prepared checkpoint (manifest.json; models/wan2.1/prepare.py) over the published checkpoint at
//! WAN21_CHECKPOINT_DIR, loaded by core_infra::model_load.
use serde_json::{json, Value};
use std::path::Path;
use wan21_native::cuda::{Buffer, Device};
use wan21_native::dit::{Dit, Grid, DIM};
use wan21_native::gold::{compare, Gold};
use wan21_native::sampler::UniPc;
use wan21_native::Result;

/// How a native buffer relates to its gold tensor.
#[derive(Clone, Copy)]
enum Conv { Same, GoldToBf16, OursToF32 }

struct Case<'a> { gold: &'a Gold, device: &'a Device }
impl Case<'_> {
    fn upload(&self, name: &str) -> Result<Buffer> { Buffer::upload(&self.gold.bytes(name)?, self.device) }
    fn compare_at(&self, label: &str, got: u64, want: &str, conv: Conv) -> Result<Value> {
        let entry = self.gold.entry(want)?;
        let elements = entry.shape.iter().product::<usize>();
        let (got_bytes, dtype) = match conv {
            Conv::Same => (entry.bytes, entry.dtype.clone()),
            Conv::GoldToBf16 => (elements * 2, "torch.bfloat16".to_string()),
            Conv::OursToF32 => (elements * 2, entry.dtype.clone()),
        };
        let staging = Buffer::new(got_bytes)?;
        staging.copy_from_device(got, got_bytes, self.device)?;
        let mut ours = staging.download(self.device)?;
        if entry.file.is_none() {
            let equal = matches!(conv, Conv::Same) && wan21_native::gold::sha256_hex(&ours) == entry.sha256;
            return Ok(json!({"case": label, "want": want, "dtype": entry.dtype, "shape": entry.shape,
                             "diff": {"sha256_equal": equal, "hash_only": true}}));
        }
        let mut want_bytes = self.gold.bytes(want)?;
        match conv {
            Conv::Same => {}
            Conv::GoldToBf16 => want_bytes = f32_to_bf16(&want_bytes),
            Conv::OursToF32 => ours = bf16_to_f32(&ours),
        }
        let diff = compare(&ours, &want_bytes, &dtype)?;
        Ok(json!({"case": label, "want": want, "dtype": dtype, "shape": entry.shape, "diff": diff}))
    }
}

fn f32_to_bf16(bytes: &[u8]) -> Vec<u8> {
    bytes.chunks_exact(4).flat_map(|b| {
        let bits = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let rounded = (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
        rounded.to_le_bytes()
    }).collect()
}
fn bf16_to_f32(bytes: &[u8]) -> Vec<u8> {
    bytes.chunks_exact(2).flat_map(|b| ((u16::from_le_bytes([b[0], b[1]]) as u32) << 16).to_le_bytes()).collect()
}

fn report(check: &str, results: Vec<Value>) -> Value {
    let equal = results.iter().filter(|r| r["diff"]["sha256_equal"] == true).count();
    json!({"check": check, "passed": equal == results.len(), "equal": equal, "total": results.len(), "results": results})
}

/// The first DiT forward of a fixture gold, from its recorded inputs: context embeddings, time
/// embedding, every block-0 operator, every block output, the head and the prediction.
fn forward(root: &str, weights: &str) -> Result<Value> {
    let gold = Gold::open(root)?;
    let device = Device::open()?;
    let case = Case { gold: &gold, device: &device };
    let f = "dit/f0000";
    let shape = gold.entry(&format!("{f}/in/x"))?.shape.clone();
    let grid = Grid { frames: shape[1], height: shape[2], width: shape[3] };
    let t0 = std::time::Instant::now();
    let mut dit = Dit::load(&Path::new(weights).join("dit.safetensors"), grid, &device)?;
    dit.enable_debug()?;
    let load_seconds = t0.elapsed().as_secs_f64();
    let (x, y, context, clip) = (case.upload(&format!("{f}/in/x"))?, case.upload(&format!("{f}/in/y"))?,
                                 case.upload(&format!("{f}/in/context"))?, case.upload(&format!("{f}/in/clip_fea"))?);
    let rows = gold.entry(&format!("{f}/in/context"))?.shape[0];
    let t_bytes = gold.bytes(&format!("{f}/in/t"))?;
    let t = i64::from_le_bytes(t_bytes[..8].try_into()?);
    let mut results = Vec::new();
    let fx = |name: &str| format!("{f}/fx/{name}");
    let ctx = dit.pass_context(context.dev(), rows, clip.dev())?;
    results.push(case.compare_at("text_embedding", ctx.text.dev(), &fx("text_embedding/out"), Conv::Same)?);
    results.push(case.compare_at("img_emb", ctx.image_f32.dev(), &fx("img_emb/out"), Conv::Same)?);
    let [k, v, k_img, v_img] = Dit::context_kv(&ctx, 0);
    results.push(case.compare_at("b0 cross k (normed)", k, &fx("blocks.0.cross_attn.norm_k/out"), Conv::Same)?);
    results.push(case.compare_at("b0 cross v", v, &fx("blocks.0.cross_attn.v/out"), Conv::Same)?);
    results.push(case.compare_at("b0 cross k_img (normed)", k_img, &fx("blocks.0.cross_attn.norm_k_img/out"), Conv::Same)?);
    results.push(case.compare_at("b0 cross v_img", v_img, &fx("blocks.0.cross_attn.v_img/out"), Conv::Same)?);
    dit.time(t)?;
    let (e, e0) = dit.time_buffers();
    results.push(case.compare_at("time_embedding", e, &fx("time_embedding/out"), Conv::Same)?);
    results.push(case.compare_at("time_projection", e0, &fx("time_projection/out"), Conv::Same)?);
    let out = Buffer::new(grid.latent_elements(16) * 4)?;
    let map: &[(&str, &str, Conv)] = &[
        ("patch_embedding/out", "patch_embedding/out", Conv::Same),
        ("b0/xm_self", "blocks.0.self_attn.q/in", Conv::GoldToBf16),
        ("b0/q", "blocks.0.self_attn.q/out", Conv::Same), ("b0/k", "blocks.0.self_attn.k/out", Conv::Same),
        ("b0/v", "blocks.0.self_attn.v/out", Conv::Same),
        ("b0/qn", "blocks.0.self_attn.norm_q/out", Conv::Same), ("b0/kn", "blocks.0.self_attn.norm_k/out", Conv::Same),
        ("b0/qr_f32", "rope0/out", Conv::Same), ("b0/kr_f32", "rope1/out", Conv::Same),
        ("b0/attn", "attn0/out", Conv::OursToF32), ("b0/o", "blocks.0.self_attn.o/out", Conv::Same),
        ("b0/x1", "blocks.0.norm3/in", Conv::Same), ("b0/xm_cross", "blocks.0.cross_attn.q/in", Conv::GoldToBf16),
        ("b0/cq", "blocks.0.cross_attn.q/out", Conv::Same), ("b0/cqn", "blocks.0.cross_attn.norm_q/out", Conv::Same),
        ("b0/img", "attn1/out", Conv::Same), ("b0/txt", "attn2/out", Conv::Same),
        ("b0/sum", "blocks.0.cross_attn.o/in", Conv::Same), ("b0/co", "blocks.0.cross_attn.o/out", Conv::Same),
        ("b0/x2", "blocks.0.norm2/in", Conv::Same), ("b0/xm_ffn", "blocks.0.ffn.0/in", Conv::GoldToBf16),
        ("b0/h", "blocks.0.ffn.0/out", Conv::Same), ("b0/g", "blocks.0.ffn.1/out", Conv::Same),
        ("b0/f", "blocks.0.ffn.2/out", Conv::Same), ("head/out", "head/out", Conv::Same),
    ];
    let t1 = std::time::Instant::now();
    dit.forward(x.dev(), y.dev(), &ctx, out.dev(), &mut |name, addr| {
        let (want, conv) = match map.iter().find(|(n, _, _)| *n == name) {
            Some((_, w, c)) => (fx(w), *c),
            None if name.starts_with("blocks.") => (fx(&name.replace("/out", "/out")), Conv::Same),
            None => return Ok(()),
        };
        if gold.tensors.contains_key(&want) { results.push(case.compare_at(name, addr, &want, conv)?); }
        Ok(())
    })?;
    device.sync()?;
    let forward_seconds = t1.elapsed().as_secs_f64();
    results.push(case.compare_at("prediction", out.dev(), &format!("{f}/out"), Conv::Same)?);
    let mut r = report("forward", results);
    r["seconds"] = json!({"load": load_seconds, "forward_with_checks": forward_seconds});
    r["grid"] = json!([grid.frames, grid.height, grid.width]);
    r["hidden"] = json!(DIM);
    Ok(r)
}

/// The scheduler rung: the schedule, then every step's guidance and UniPC update from the gold's own
/// per-step inputs (the native state carries over between steps like the reference's).
fn sched(root: &str) -> Result<Value> {
    let gold = Gold::open(root)?;
    let device = Device::open()?;
    let case = Case { gold: &gold, device: &device };
    let steps = gold.entry("sched/timesteps")?.shape[0];
    let shift = gold.meta["sched/set_timesteps"]["kwargs"]["shift"].as_str().ok_or("shift")?.parse::<f64>()?;
    let n: usize = gold.entry("sched/step00/sample")?.shape.iter().product();
    let mut unipc = UniPc::new(steps, shift, n)?;
    let mut results = Vec::new();
    let host = |name: &str, bytes: Vec<u8>| -> Result<Value> {
        let entry = gold.entry(name)?;
        Ok(json!({"case": name, "diff": compare(&bytes, &gold.bytes(name)?, &entry.dtype)?}))
    };
    results.push(host("sched/sigmas", unipc.sigmas.iter().flat_map(|v| v.to_le_bytes()).collect())?);
    results.push(host("sched/timesteps", unipc.timesteps.iter().flat_map(|v| v.to_le_bytes()).collect())?);
    let (guided, prev) = (Buffer::new(n * 4)?, Buffer::new(n * 4)?);
    for step in 0..steps {
        let name = |s: &str| format!("sched/step{step:02}/{s}");
        let (cond, uncond) = (format!("dit/f{:04}/out", 2 * step), format!("dit/f{:04}/out", 2 * step + 1));
        if gold.tensors.get(&cond).is_some_and(|e| e.file.is_some()) {
            let (c, u) = (case.upload(&cond)?, case.upload(&uncond)?);
            let guide = 5.0;
            unipc.guidance(c.dev(), u.dev(), guided.dev(), guide, device.stream)?;
            results.push(case.compare_at(&format!("step {step} guidance"), guided.dev(), &name("model_output"), Conv::Same)?);
        }
        let (mo, sample) = (case.upload(&name("model_output"))?, case.upload(&name("sample"))?);
        unipc.step(mo.dev(), sample.dev(), prev.dev(), &device, &mut |what, addr| {
            results.push(case.compare_at(&format!("step {step} {what}"), addr, &name(what), Conv::Same)?);
            Ok(())
        })?;
        results.push(case.compare_at(&format!("step {step} prev_sample"), prev.dev(), &name("prev_sample"), Conv::Same)?);
    }
    Ok(report("sched", results))
}

/// T5 on the gold's token ids (prompt and negative prompt, batch 1 each as the reference) and the CLIP
/// visual encoder on the gold's image tensor.
fn encoders(root: &str, weights: &str) -> Result<Value> {
    use wan21_native::clip::ClipVisual;
    use wan21_native::t5::T5;
    use wan21_native::weights::SafeTensors;
    let gold = Gold::open(root)?;
    let device = Device::open()?;
    let case = Case { gold: &gold, device: &device };
    let w = Path::new(weights);
    let mut results = Vec::new();
    let t5 = T5::load(&SafeTensors::open(w.join("t5.safetensors"))?, &device)?;
    let out = Buffer::new(512 * 4096 * 2)?;
    for call in 0..2 {
        let (ids, mask) = (case.upload(&format!("t5/call{call}/ids"))?, case.upload(&format!("t5/call{call}/mask"))?);
        t5.forward(ids.dev(), mask.dev(), 1, out.dev())?;
        results.push(case.compare_at(&format!("t5 call {call}"), out.dev(), &format!("t5/call{call}/context"), Conv::Same)?);
    }
    drop(t5);
    let clip = ClipVisual::load(&SafeTensors::open(w.join("clip.safetensors"))?, &device)?;
    let shape = gold.entry("clip/in/video")?.shape.clone();
    let video = case.upload("clip/in/video")?;
    let feats = Buffer::new(257 * 1280 * 2)?;
    clip.forward(video.dev(), shape[2], shape[3], feats.dev())?;
    results.push(case.compare_at("clip visual", feats.dev(), "clip/out", Conv::Same)?);
    Ok(report("encoders", results))
}

/// The fp32 VAE: encode of the conditioning video and decode of the final latent, from the gold's
/// own inputs.
fn vae(root: &str, weights: &str) -> Result<Value> {
    use wan21_native::vae::{Vae, VTensor};
    use wan21_native::weights::SafeTensors;
    let gold = Gold::open(root)?;
    let device = Device::open()?;
    let case = Case { gold: &gold, device: &device };
    let vae = Vae::load(&SafeTensors::open(Path::new(weights).join("vae.safetensors"))?, &device)?;
    let mut results = Vec::new();
    let s = gold.entry("vae/encode/in")?.shape.clone();
    let video = VTensor { buf: case.upload("vae/encode/in")?, c: s[0], t: s[1], h: s[2], w: s[3], tc: false };
    let t0 = std::time::Instant::now();
    let latent = vae.encode(&video)?;
    device.sync()?;
    let encode_seconds = t0.elapsed().as_secs_f64();
    results.push(case.compare_at("vae encode", latent.dev(), "vae/encode/out", Conv::Same)?);
    let z = gold.entry("vae/decode/in")?.shape.clone();
    let zin = case.upload("vae/decode/in")?;
    let t1 = std::time::Instant::now();
    let (videos, _) = vae.decode(zin.dev(), z[1], z[2], z[3])?;
    device.sync()?;
    let decode_seconds = t1.elapsed().as_secs_f64();
    results.push(case.compare_at("vae decode", videos.dev(), "vae/decode/out", Conv::Same)?);
    let mut r = report("vae", results);
    r["seconds"] = json!({"encode": encode_seconds, "decode": decode_seconds});
    Ok(r)
}


const DEFAULT_CHECKPOINT: &str =
    "/models/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-720P/snapshots/8823af45fcc58a8aa999a54b04be9abc7d2aac98";
fn checkpoint_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("WAN21_CHECKPOINT_DIR").unwrap_or_else(|_| DEFAULT_CHECKPOINT.to_string()))
}
fn is_prepared(weights: &str) -> bool { Path::new(weights).join("manifest.json").exists() }
fn tokenizer_of(weights: &str) -> std::path::PathBuf {
    if is_prepared(weights) { checkpoint_dir().join("google/umt5-xxl/tokenizer.json") } else { Path::new(weights).join("tokenizer.json") }
}

/// The streaming in-place load of a prepared checkpoint (readers started before the CUDA context),
/// then every loaded tensor's device bytes against a converted weight export.
fn load(prepared: &str, export: &str) -> Result<Value> {
    use wan21_native::weights::SafeTensors;
    let t0 = std::time::Instant::now();
    let pending = wan21_native::pipeline::start_prepared(Path::new(prepared), &checkpoint_dir())?;
    let readers = t0.elapsed().as_secs_f64();
    let device = Device::open()?;
    let context = t0.elapsed().as_secs_f64();
    let loaded = unsafe { pending.finish(device.stream)? };
    let total = t0.elapsed().as_secs_f64();
    let (mut compared, mut equal, mut not_loaded, mut mismatches) = (0usize, 0usize, 0usize, Vec::new());
    for component in ["dit", "t5", "clip", "vae"] {
        let st = SafeTensors::open(Path::new(export).join(format!("{component}.safetensors")))?;
        let mut names: Vec<&String> = st.entries.keys().collect();
        names.sort();
        for name in names {
            let key = format!("{component}.{name}");
            let Some(view) = loaded.tensors.get(&key) else { not_loaded += 1; continue; };
            let got = Buffer::view(view.ptr, view.bytes)?.download(&device)?;
            compared += 1;
            if got == st.bytes(name)? && view.dtype.name() == st.entries[name].dtype { equal += 1; }
            else if mismatches.len() < 20 { mismatches.push(json!({"tensor": key, "dtype": view.dtype.name(), "export_dtype": st.entries[name].dtype})); }
        }
    }
    Ok(json!({"check": "load", "passed": compared > 0 && equal == compared, "compared": compared, "equal": equal,
              "export_tensors_not_loaded": not_loaded, "mismatches": mismatches, "stats": loaded.stats,
              "seconds": {"readers_started": readers, "cuda_context": context, "loaded": total}}))
}

/// The gold's raw request files: `<gold>/<run>/../inputs`.
fn inputs_of(root: &str) -> Result<std::path::PathBuf> {
    Ok(Path::new(root).parent().ok_or("gold run has no parent")?.join("inputs"))
}

fn f32_bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn i64_bytes(v: &[i64]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

/// The CPU front end from the raw request files: the image tensors, the latent grid (against the gold's
/// noise shape), both token sequences, and the bicubic conditioning video on the device.
fn frontend(root: &str, weights: &str) -> Result<Value> {
    use wan21_native::frontend::prepare;
    use wan21_native::vae::VTensor;
    let gold = Gold::open(root)?;
    let device = Device::open()?;
    let case = Case { gold: &gold, device: &device };
    let prepared = prepare(&inputs_of(root)?, &tokenizer_of(weights))?;
    let mut results = Vec::new();
    let host = |label: &str, bytes: Vec<u8>, want: &str| -> Result<Value> {
        let entry = gold.entry(want)?;
        let diff = match entry.file {
            Some(_) => serde_json::to_value(compare(&bytes, &gold.bytes(want)?, &entry.dtype)?)?,
            None => json!({"sha256_equal": wan21_native::gold::sha256_hex(&bytes) == entry.sha256, "hash_only": true}),
        };
        Ok(json!({"case": label, "want": want, "diff": diff}))
    };
    results.push(host("to_tensor", f32_bytes(&prepared.to_tensor), "image/to_tensor")?);
    results.push(host("normalized image", f32_bytes(&prepared.image), "clip/in/video")?);
    for (call, (ids, mask, _)) in prepared.tokens.iter().enumerate() {
        results.push(host(&format!("t5 call {call} ids"), i64_bytes(ids), &format!("t5/call{call}/ids"))?);
        results.push(host(&format!("t5 call {call} mask"), i64_bytes(mask), &format!("t5/call{call}/mask"))?);
    }
    let noise = gold.entry("noise/randn0")?.shape.clone();
    let grid_equal = noise[1] == prepared.latent_frames() && noise[2] == prepared.lat_h && noise[3] == prepared.lat_w;
    results.push(json!({"case": "latent grid", "want": "noise/randn0 shape", "diff": {"sha256_equal": grid_equal},
                        "ours": [prepared.latent_frames(), prepared.lat_h, prepared.lat_w], "gold": noise}));
    let want = gold.entry("vae/encode/in")?.shape.clone();
    let (h, w) = (prepared.height(), prepared.width());
    let video = VTensor::new(3, want[1], h, w)?;
    video.buf.zero(&device)?;
    let image = Buffer::upload(&f32_bytes(&prepared.image), &device)?;
    let up = |b: Vec<u8>| Buffer::upload(&b, &device);
    let (iy, wy) = (up(prepared.taps_y.0.iter().flat_map(|x| x.to_le_bytes()).collect())?, up(f32_bytes(&prepared.taps_y.1))?);
    let (ix, wx) = (up(prepared.taps_x.0.iter().flat_map(|x| x.to_le_bytes()).collect())?, up(f32_bytes(&prepared.taps_x.1))?);
    wan21_native::pipeline::bicubic(image.dev(), prepared.src_h, prepared.src_w, iy.dev(), wy.dev(), ix.dev(), wx.dev(),
                                    video.dev(), h, w, want[1], &device)?;
    results.push(case.compare_at("vae input (bicubic + zeros)", video.dev(), "vae/encode/in", Conv::Same)?);
    let mut r = report("frontend", results);
    r["tokens"] = json!([prepared.tokens[0].2, prepared.tokens[1].2]);
    r["source"] = json!([prepared.src_h, prepared.src_w]);
    Ok(r)
}

/// The whole request from the raw files, every recorded tensor compared as it is produced. With
/// WAN_REPEATS=n, n more unobserved runs on the loaded models time the warm pipeline.
fn e2e(root: &str, weights: &str) -> Result<Value> {
    use wan21_native::frontend::prepare;
    use wan21_native::pipeline::{generate, grid, Models};
    let gold = Gold::open(root)?;
    let t1 = std::time::Instant::now();
    // Prepared checkpoints stream in place: readers start before the CUDA context and the front end.
    let pending = if is_prepared(weights) { Some(wan21_native::pipeline::start_prepared(Path::new(weights), &checkpoint_dir())?) }
                  else { None };
    let device = Device::open()?;
    let case = Case { gold: &gold, device: &device };
    let t0 = std::time::Instant::now();
    let prepared = prepare(&inputs_of(root)?, &tokenizer_of(weights))?;
    let frontend_seconds = t0.elapsed().as_secs_f64();
    let mut models = match pending {
        Some(pending) => Models::from_loaded(unsafe { pending.finish(device.stream)? }, grid(&prepared), &device)?,
        None => Models::load(Path::new(weights), grid(&prepared), &device)?,
    };
    let load_seconds = t1.elapsed().as_secs_f64();
    let mut results = Vec::new();
    let generated = generate(&mut models, &prepared, &mut |name, addr| {
        if gold.tensors.contains_key(name) { results.push(case.compare_at(name, addr, name, Conv::Same)?); }
        Ok(())
    })?;
    let recorded = gold.tensors.keys().filter(|k| !k.starts_with("image/") && !k.starts_with("sched/sigmas")
                                                  && !k.starts_with("sched/timesteps") && !k.contains("/fx/")).count();
    let mut warm = Vec::new();
    let repeats: usize = std::env::var("WAN_REPEATS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    for _ in 0..repeats {
        let t = std::time::Instant::now();
        let prepared = prepare(&inputs_of(root)?, &tokenizer_of(weights))?;
        let front = t.elapsed().as_secs_f64();
        let g = generate(&mut models, &prepared, &mut |_, _| Ok(()))?;
        let digest = wan21_native::gold::sha256_hex(&g.video.download(&device)?);
        warm.push(json!({"frontend": front, "timings": g.timings, "video_sha256": digest,
                         "video_equal": gold.entry("video/out").map(|e| e.sha256 == digest).unwrap_or(false)}));
    }
    let latents = results.iter().filter(|r| {
        let w = r["want"].as_str().unwrap_or("");
        w.starts_with("dit/") || w.starts_with("sched/") || w.starts_with("noise/") || w == "vae/encode/out"
    }).count();
    let mut r = report("e2e", results);
    r["compared"] = json!({"tensors": r["total"], "latent_tensors": latents, "recorded_in_gold": recorded});
    r["seconds"] = json!({"frontend": frontend_seconds, "load": load_seconds, "generate_with_checks": generated.timings,
                          "warm": warm});
    r["video"] = json!({"frames": generated.frames, "height": prepared.height(), "width": prepared.width()});
    Ok(r)
}

extern "C" {
    fn cudaProfilerStart() -> std::ffi::c_int;
    fn cudaProfilerStop() -> std::ffi::c_int;
}

/// DiT forwards (time embedding + forward, as one reference model call) on a gold run's first forward
/// inputs: 2 warm-up, then 3 between cudaProfilerStart/Stop (the Nsight capture range), each
/// wall-timed after a synchronize. The per-pass cross-attention keys/values (computed once per
/// request natively, in every forward by the reference) are timed separately.
fn profile(root: &str, weights: &str) -> Result<Value> {
    let gold = Gold::open(root)?;
    let device = Device::open()?;
    let case = Case { gold: &gold, device: &device };
    let f = "dit/f0000";
    let shape = gold.entry(&format!("{f}/in/x"))?.shape.clone();
    let grid = Grid { frames: shape[1], height: shape[2], width: shape[3] };
    let dit = Dit::load(&Path::new(weights).join("dit.safetensors"), grid, &device)?;
    let (x, y, context, clip) = (case.upload(&format!("{f}/in/x"))?, case.upload(&format!("{f}/in/y"))?,
                                 case.upload(&format!("{f}/in/context"))?, case.upload(&format!("{f}/in/clip_fea"))?);
    let rows = gold.entry(&format!("{f}/in/context"))?.shape[0];
    let t = i64::from_le_bytes(gold.bytes(&format!("{f}/in/t"))?[..8].try_into()?);
    let mut pass_context = Vec::new();
    let mut ctx = None;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        ctx = Some(dit.pass_context(context.dev(), rows, clip.dev())?);
        device.sync()?;
        pass_context.push(t0.elapsed().as_secs_f64());
    }
    let ctx = ctx.unwrap();
    let out = Buffer::new(grid.latent_elements(16) * 4)?;
    let forward = || -> Result<f64> {
        let t0 = std::time::Instant::now();
        dit.time(t)?;
        dit.forward(x.dev(), y.dev(), &ctx, out.dev(), &mut |_, _| Ok(()))?;
        device.sync()?;
        Ok(t0.elapsed().as_secs_f64())
    };
    for _ in 0..2 { forward()?; }
    unsafe { cudaProfilerStart() };
    let walls = (0..3).map(|_| forward()).collect::<Result<Vec<_>>>()?;
    unsafe { cudaProfilerStop() };
    let check = case.compare_at("prediction", out.dev(), &format!("{f}/out"), Conv::Same)?;
    Ok(json!({"check": "profile", "forward_wall_s": walls, "pass_context_s": pass_context,
              "output_equals_gold": check["diff"]["sha256_equal"], "grid": [grid.frames, grid.height, grid.width]}))
}

extern "C" { fn cudaDeviceSynchronize() -> std::ffi::c_int; }

/// Isolate the framework's immutable pinned-RAM transfer from file reads,
/// verification, context creation and model construction. This diagnostic
/// copies stored dtypes (including CLIP F32); it does not time dtype conversion
/// or claim to implement a resident Wan model-switch lifecycle.
fn ramcopy(prepared:&str,repeats:&str)->Result<Value>{
    use native_core_infra::{checkpoint::Manifest,model_load::{self,Options},weight_loader::tensor_allocation_layout};
    use sha2::Digest;
    use std::{collections::HashMap,time::Instant};
    let repeats:usize=repeats.parse()?;
    if !(1..=20).contains(&repeats){return Err("ramcopy repeats must be 1..20".into());}
    let pin=std::env::var("WAN21_MANIFEST_SHA256").ok();
    let manifest=Manifest::load(&Path::new(prepared).join("manifest.json"),pin.as_deref())?;
    let roots=HashMap::from([("checkpoint".into(),checkpoint_dir()),("prepared".into(),prepared.into())]);
    let mut requests=wan21_native::pipeline::prepared_requests(&manifest);
    for request in &mut requests{request.dtype=manifest.tensor(&request.name)?.dtype;}
    let sizes=requests.iter().map(|r|manifest.tensor(&r.name).map(|t|t.bytes)).collect::<std::result::Result<Vec<_>,_>>()?;
    let (offsets,allocation_bytes)=tensor_allocation_layout(&sizes)?;
    let bytes=sizes.iter().sum::<usize>();
    let mut options=Options::from_env()?;options.retained_budget=64usize<<30;
    if !options.direct{return Err("ramcopy requires the existing direct pinned source policy".into());}
    let preparation=Instant::now();
    let pending=model_load::start(&manifest,&roots,&requests,options)?;
    let device=Device::open()?;
    let host=pending.finish_host()?;
    let preparation_seconds=preparation.elapsed().as_secs_f64();
    let mut rows=Vec::new();let mut verified=0usize;
    for index in 0..repeats{
        let begun=Instant::now();
        let buffer=Buffer::new(allocation_bytes)?;device.sync()?;
        let allocation_seconds=begun.elapsed().as_secs_f64();
        let jobs:Vec<_>=requests.iter().zip(&offsets).map(|(r,&offset)|(r.name.as_str(),buffer.dev()+offset as u64)).collect();
        let copy=Instant::now();unsafe{host.upload(&jobs,device.stream)}?;device.sync()?;
        let copy_seconds=copy.elapsed().as_secs_f64();
        rows.push(json!({"index":index,"bytes":bytes,"tensors":requests.len(),"allocation_seconds":allocation_seconds,
            "copy_and_wait_seconds":copy_seconds,"allocation_and_copy_seconds":begun.elapsed().as_secs_f64(),
            "copy_gb_per_second":bytes as f64/1e9/copy_seconds}));
        // Validate the actual destination, outside the timed copy. The trusted
        // tensor guards cover the same original bytes admitted by the loader.
        if index==repeats-1{
            for ((request,&offset),&size) in requests.iter().zip(&offsets).zip(&sizes){
                let tensor=manifest.tensor(&request.name)?;
                let guard=&tensor.guard;let count=size.min(1024);
                for (at,want) in [(0,&guard.head_sha256),(size.saturating_sub(count),&guard.tail_sha256)]{
                    let actual=Buffer::view(buffer.dev()+offset as u64+at as u64,count)?.download(&device)?;
                    let actual=format!("{:x}",sha2::Sha256::digest(&actual));
                    if actual!=*want{return Err(format!("RAM copy guard mismatch: {}",request.name).into());}
                }
                verified+=1;
            }
        }
    }
    Ok(json!({"check":"ramcopy","passed":verified==requests.len(),"stored_bytes":bytes,"allocation_bytes":allocation_bytes,
        "retained_host_bytes":host.bytes(),"host_preparation_seconds_excluded":preparation_seconds,
        "scope":"original stored tensors; no runtime dtype conversion or model construction",
        "verification":"destination head/tail SHA256 guards, outside timing","verified_tensors":verified,"samples":rows}))
}

/// Cold start of one request with weights streamed from a prepared checkpoint, through the same
/// schedule the resident server uses (pipeline::cold_start): UNIX wall-clock stamps from
/// WAN_CLOCK_START (the caller's clock origin) through readers, CUDA context, the front end,
/// the encoders' weights and run, the DiT's weights, to the first DiT forward (stamps of GPU work
/// follow a device synchronize). WAN_COLDSTART_FULL=1 runs the whole request; WAN_PROFILE_FIRST=1
/// makes the first DiT forward the Nsight capture range.
fn coldstart(inputs: &str, weights: &str) -> Result<Value> {
    use wan21_native::frontend::prepare;
    use wan21_native::pipeline::{cold_start, unix_now, Marks};
    let marks = Marks::default();
    if let Some(t) = std::env::var("WAN_CLOCK_START").ok().and_then(|v| v.parse::<f64>().ok()) {
        marks.borrow_mut().push(("clock_start".into(), t));
    }
    marks.borrow_mut().push(("main".into(), unix_now()));
    let full = std::env::var("WAN_COLDSTART_FULL").as_deref() == Ok("1");
    let profile_first = std::env::var("WAN_PROFILE_FIRST").as_deref() == Ok("1");
    // Diagnostic only: validate this exact cold schedule against the recorded
    // tensor hashes. Never enable this extra synchronization in timing runs.
    let gold = std::env::var("WAN_COLDSTART_GOLD").ok().map(Gold::open).transpose()?;
    let mut check_device = None;
    let mut comparisons = Vec::new();
    const STOP: &str = "coldstart: first DiT forward reached";
    let (inputs_dir, tokenizer) = (Path::new(inputs).to_path_buf(), tokenizer_of(weights));
    let mut observe = |name: &str, address: u64| -> Result<()> {
        if let Some(gold) = gold.as_ref().filter(|g| g.tensors.contains_key(name)) {
            if unsafe { cudaDeviceSynchronize() } != 0 { return Err("validation synchronization failed".into()); }
            if check_device.is_none() { check_device = Some(Device::open()?); }
            let case = Case { gold, device: check_device.as_ref().unwrap() };
            comparisons.push(case.compare_at(name, address, name, Conv::Same)?);
        }
        if matches!(name, "clip/out" | "vae/encode/out" | "noise/randn0" | "dit/f0000/in/x" | "dit/f0000/out" | "dit/f0001/out"
                          | "video/out") {
            if unsafe { cudaDeviceSynchronize() } != 0 { return Err("cudaDeviceSynchronize failed".into()); }
            marks.borrow_mut().push((name.to_string(), unix_now()));
        }
        if profile_first && name == "dit/f0000/in/x" { unsafe { cudaProfilerStart() }; }
        if profile_first && name == "dit/f0000/out" { unsafe { cudaProfilerStop() }; }
        if name == "dit/f0000/out" && !full { return Err(STOP.into()); }
        Ok(())
    };
    match cold_start(Path::new(weights), &checkpoint_dir(), move || prepare(&inputs_dir, &tokenizer).map_err(|e| e.to_string()),
                     &marks, &mut observe) {
        Ok(_) => {}
        Err(e) if e.to_string() == STOP => {}
        Err(e) => return Err(e),
    }
    let marks = marks.into_inner();
    let origin = marks[0].1;
    let phases: serde_json::Map<String, Value> = marks.iter().map(|(k, t)| (k.clone(), json!(t - origin))).collect();
    let equal = comparisons.iter().filter(|r| r["diff"]["sha256_equal"] == true).count();
    Ok(json!({"check": "coldstart", "passed": equal == comparisons.len(), "origin": marks[0].0,
              "seconds_since_origin": phases, "full": full,
              "validation": {"enabled": gold.is_some(), "equal": equal, "total": comparisons.len(), "results": comparisons}}))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("forward") if args.len() == 4 => forward(&args[2], &args[3]),
        Some("sched") if args.len() == 3 => sched(&args[2]),
        Some("encoders") if args.len() == 4 => encoders(&args[2], &args[3]),
        Some("vae") if args.len() == 4 => vae(&args[2], &args[3]),
        Some("frontend") if args.len() == 4 => frontend(&args[2], &args[3]),
        Some("e2e") if args.len() == 4 => e2e(&args[2], &args[3]),
        Some("profile") if args.len() == 4 => profile(&args[2], &args[3]),
        Some("load") if args.len() == 4 => load(&args[2], &args[3]),
        Some("coldstart") if args.len() == 4 => coldstart(&args[2], &args[3]),
        Some("ramcopy") if args.len() == 4 => ramcopy(&args[2], &args[3]),
        _ => Err("usage: wan21-playground forward|encoders|vae|frontend|e2e <gold run> <weights dir> | sched <gold run>".into()),
    };
    match result {
        Ok(v) => println!("{}", serde_json::to_string_pretty(&v).unwrap()),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}
