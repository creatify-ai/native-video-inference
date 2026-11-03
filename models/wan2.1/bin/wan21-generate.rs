//! Official Wan 2.1 image-to-video (I2V-14B-720P) on one GPU, natively: the same video as
//! `python generate.py --task i2v-14B --size 1280*720 ...` with the DiT in bf16 (bitwise identical
//! latents on B200; see README.md).
//!
//!   wan21-generate --weights <dir> [--checkpoint <dir>] --image in.png --prompt "..." [--out out.mp4]
//!                  [--seed 42] [--steps 40] [--shift 5.0] [--guide-scale 5.0] [--frames 81]
//!                  [--max-area 921600] [--negative-prompt "..."] [--gpus 1|2] [--repeat N]
//!
//! `--weights` is a prepared checkpoint (manifest.json from core_infra/runtime/checkpoint.py) over the
//! published Wan2.1-I2V-14B-720P checkpoint given by `--checkpoint`, streamed in place; or a
//! directory of converted weights for explicit `--diagnostic-converted` parity tooling. The MP4 is
//! written through `ffmpeg` (libx264, CRF 10 as imageio's quality=8, 16 fps as cache_video).
//!
//! `--gpus 2` runs every unconditional DiT pass on a second GPU. `--repeat N` serves the request N
//! times in one process as wan21-server does: later requests reuse the resident models and the first
//! request's conditioning (same prompts and image).
use std::path::PathBuf;
use std::process::{Command, Stdio};
use wan21_native::cuda::Device;
use wan21_native::frontend::{prepare_request, Request, NEGATIVE_PROMPT};
use wan21_native::pipeline::{cold_start_on, generate, generate_request, grid, write_mp4, Marks, Models};
use wan21_native::Result;


fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut weights, mut checkpoint, mut image, mut out) = (None, None::<PathBuf>, None, PathBuf::from("out.mp4"));
    let mut diagnostic_converted = false;
    let (mut gpus, mut repeat) = (1usize, 1usize);
    let mut request = Request { prompt: String::new(), negative_prompt: NEGATIVE_PROMPT.to_string(), max_area: 720.0 * 1280.0,
                                frame_num: 81, sample_steps: 40, shift: 5.0, guide_scale: 5.0, seed: 42 };
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--diagnostic-converted" => diagnostic_converted = true,
            "--weights" => weights = Some(PathBuf::from(value()?)),
            "--checkpoint" => checkpoint = Some(PathBuf::from(value()?)),
            "--image" => image = Some(PathBuf::from(value()?)),
            "--out" => out = PathBuf::from(value()?),
            "--prompt" => request.prompt = value()?,
            "--negative-prompt" => request.negative_prompt = value()?,
            "--seed" => request.seed = value()?.parse()?,
            "--steps" => request.sample_steps = value()?.parse()?,
            "--shift" => request.shift = value()?.parse()?,
            "--guide-scale" => request.guide_scale = value()?.parse()?,
            "--frames" => request.frame_num = value()?.parse()?,
            "--max-area" => request.max_area = value()?.parse()?,
            "--gpus" => gpus = value()?.parse()?,
            "--repeat" => repeat = value()?.parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let weights = weights.ok_or("--weights is required")?;
    let image = image.ok_or("--image is required")?;
    if request.prompt.is_empty() { return Err("--prompt is required".into()); }
    if request.frame_num % 4 != 1 { return Err("--frames must be 4n+1".into()); }
    if !matches!(gpus, 1 | 2) { return Err("--gpus must be 1 or 2".into()); }
    if repeat < 1 { return Err("--repeat must be at least 1".into()); }
    let prepared_checkpoint = weights.join("manifest.json").is_file();
    if !prepared_checkpoint && !diagnostic_converted {
        return Err("Wan generation requires a prepared checkpoint manifest and its trusted SHA-256; use --diagnostic-converted only for converted-weight parity tooling".into());
    }
    if prepared_checkpoint && diagnostic_converted {
        return Err("--diagnostic-converted requires converted weights without a prepared checkpoint manifest".into());
    }
    // Fail before the long run if the video cannot be written.
    if !Command::new("ffmpeg").arg("-version").stdout(Stdio::null()).status().is_ok_and(|s| s.success()) {
        return Err("ffmpeg is required to write the MP4".into());
    }
    let png = std::fs::read(&image)?;
    let t0 = std::time::Instant::now();
    if prepared_checkpoint {
        // A prepared checkpoint streams in place with the framework's cold-start schedule.
        let checkpoint = checkpoint.ok_or("--checkpoint is required with a prepared checkpoint")?;
        let tokenizer = checkpoint.join("google/umt5-xxl/tokenizer.json");
        let marks = Marks::default();
        let front = |request: Request, png: Vec<u8>, tokenizer: PathBuf|
            move || prepare_request(request, &png, &tokenizer).map_err(|e| e.to_string());
        let (device, mut models, prepared, generated) = cold_start_on(gpus, &weights, &checkpoint,
            front(request.clone(), png.clone(), tokenizer.clone()), &marks, &mut |_, _| Ok(()))?;
        eprintln!("generated {} frames at {}x{} in {:.1}s from process start ({:?})", generated.frames, prepared.width(),
                  prepared.height(), t0.elapsed().as_secs_f64(), generated.timings);
        write_mp4(&generated, &prepared, device, &out)?;
        models.reuse_conditioning = true;
        for n in 2..=repeat {
            let t = std::time::Instant::now();
            let (prepared, generated) = generate_request(&mut models, front(request.clone(), png.clone(), tokenizer.clone()),
                                                         &mut |_, _| Ok(()))?;
            eprintln!("generated request {n} in {:.1}s ({:?})", t.elapsed().as_secs_f64(), generated.timings);
            write_mp4(&generated, &prepared, device, &out)?;
        }
    } else {
        let prepared = prepare_request(request, &png, &weights.join("tokenizer.json"))?;
        let device = Device::open()?;
        let mut models = Models::load(&weights, grid(&prepared), &device)?;
        eprintln!("loaded in {:.1}s; generating {} frames at {}x{}", t0.elapsed().as_secs_f64(), prepared.request.frame_num,
                  prepared.width(), prepared.height());
        let generated = generate(&mut models, &prepared, &mut |_, _| Ok(()))?;
        eprintln!("generated in {:.1}s ({:?})", generated.timings.total, generated.timings);
        write_mp4(&generated, &prepared, &device, &out)?;
    }
    eprintln!("wrote {}", out.display());
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}
