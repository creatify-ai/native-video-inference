//! The resident Wan 2.1 I2V process that core_infra/runtime's NativeService owns:
//!
//!   wan21-server serve <socket>
//!
//! It binds its socket and does no GPU work until the first `generate`, which streams the prepared
//! checkpoint with the framework's cold-start schedule (pipeline::cold_start); the models then stay
//! resident for later requests. Environment: WAN21_PREPARED (prepared checkpoint: manifest.json),
//! WAN21_CHECKPOINT (the published Wan2.1-I2V-14B-720P checkpoint), WAN21_WORLD_SIZE (1, or 2 to run
//! the unconditional DiT pass on GPU 1), NATIVE_LOAD_* (loader policy).
//!
//! Commands (core_infra::resident): ["generate", "<request json>"] with {"prompt", "image" (PNG
//! path), "out" (MP4 path), optional generate() parameters as request.json} -> the output path,
//! size, timings and phase stamps (and the video's SHA-256 when WAN21_REPORT_SHA=1); ["health"].
use native_core_infra::resident::{serve, Handler};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use wan21_native::cuda::Device;
use wan21_native::frontend::{prepare_request, Request};
use wan21_native::pipeline::{cold_start_on, generate_request, unix_now, write_mp4_host, Marks, Models};
use wan21_native::Result;

struct Server { prepared: PathBuf, checkpoint: PathBuf, world: usize, state: Option<(&'static Device, Models<'static>)>, served: usize }

impl Server {
    fn generate(&mut self, spec: &str) -> Result<Value> {
        let received_monotonic_ns = native_core_infra::execution_start::monotonic_ns();
        // The pipeline's nested capture shares this request's native stream marker.
        let first_dit = native_core_infra::execution_start::Capture::begin();
        let marks = Marks::default();
        marks.borrow_mut().push(("received".into(), unix_now()));
        let spec: Value = serde_json::from_str(spec)?;
        let request = Request::from_json(&spec)?;
        let png = std::fs::read(spec["image"].as_str().ok_or("request.image")?)?;
        let out = PathBuf::from(spec["out"].as_str().ok_or("request.out")?);
        let tokenizer = self.checkpoint.join("google/umt5-xxl/tokenizer.json");
        let cold = self.state.is_none();
        // Gold observers do not establish an execution timestamp. The native
        // marker is enqueued by Dit::forward on its original CUDA stream.
        let mut observe = |_: &str, _: u64| -> Result<()> { Ok(()) };
        let (device, prepared, generated) = match self.state.as_mut() {
            None => {
                let (device, models, prepared, generated) = cold_start_on(self.world, &self.prepared, &self.checkpoint,
                    move || prepare_request(request, &png, &tokenizer).map_err(|e| e.to_string()), &marks, &mut observe)?;
                let mut models = models;
                // A request repeating the last one's prompts and image reuses its conditioning.
                models.reuse_conditioning = true;
                self.state = Some((device, models));
                (device, prepared, generated)
            }
            Some((device, models)) => {
                let (prepared,generated)=generate_request(models,
                    move||prepare_request(request,&png,&tokenizer).map_err(|e|e.to_string()),&mut observe)?;
                (*device, prepared, generated)
            }
        };
        // cold_start/generate already complete normal sampling/decode syncs.
        // report() reads the completed callback without another CUDA barrier.
        let first_dit_start = first_dit.report()?;
        marks.borrow_mut().push(("generated".into(), unix_now()));
        let video=generated.video.download_f32(device)?;
        let sha = if std::env::var("WAN21_REPORT_SHA").as_deref() == Ok("1") {
            Some(wan21_native::gold::sha256_f32(&video))
        } else { None };
        write_mp4_host(&video,generated.frames,&prepared,&out)?;
        marks.borrow_mut().push(("written".into(), unix_now()));
        self.served += 1;
        Ok(json!({"out": out, "cold": cold, "frames": generated.frames, "height": prepared.height(), "width": prepared.width(),
                  "timings": generated.timings, "marks": marks.into_inner(), "video_sha256": sha,
                  "timing_schema": "native-first-dit-v1", "request_received_monotonic_ns": received_monotonic_ns,
                  "first_dit_start": first_dit_start, "world_size": self.world,
                  "startup":self.state.as_ref().map(|(_,models)|models.startup_report())}))
    }
}

impl Handler for Server {
    fn handle(&mut self, args: &[String]) -> std::result::Result<Value, String> {
        match args.first().map(String::as_str) {
            Some("health") => Ok(json!({"loaded": self.state.is_some(), "served": self.served})),
            Some("generate") if args.len() == 2 => self.generate(&args[1]).map_err(|e| e.to_string()),
            _ => Err(format!("unknown command {args:?}")),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let run = || -> Result<()> {
        if args.len() != 3 || args[1] != "serve" { return Err("usage: wan21-server serve <socket>".into()); }
        if std::env::var_os("NATIVE_FIRST_DIT_TRACE").is_none() {
            std::env::set_var("NATIVE_FIRST_DIT_TRACE", "1");
        }
        let var = |k: &str| std::env::var(k).map(PathBuf::from).map_err(|_| format!("{k} is required"));
        let world = match std::env::var("WAN21_WORLD_SIZE").as_deref() {
            Err(_) | Ok("1") => 1, Ok("2") => 2, Ok(other) => return Err(format!("WAN21_WORLD_SIZE must be 1 or 2, not {other}").into()),
        };
        let mut server = Server { prepared: var("WAN21_PREPARED")?, checkpoint: var("WAN21_CHECKPOINT")?, world, state: None, served: 0 };
        serve(Path::new(&args[2]), &mut server)?;
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}
