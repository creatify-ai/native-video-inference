//! WanI2V.generate (wan/image2video.py) on one device, from a prepared request: T5 on the prompt and
//! the negative prompt, CLIP on the image, the VAE encode of the conditioning video (the bicubic
//! first frame and F - 1 zero frames) with the first-frame mask, seeded noise, `sample_steps` UniPC
//! steps of two DiT passes and classifier-free guidance, and the VAE decode. Every model stays
//! resident (the reference's offload_model only moves weights between devices).
use crate::clip::ClipVisual;
use crate::cuda::{check, memcpy_d2d, upload_into, Buffer, Device};
use crate::dit::{Dit, Grid, Share};
use crate::frontend::Prepared;
use crate::rng::CudaGenerator;
use crate::sampler::UniPc;
use crate::startup::{ENCODERS, VIDEO};
use crate::t5::T5;
use crate::vae::{VTensor, Vae, Z_DIM};
use crate::weights::{DeviceWeights, SafeTensors};
use crate::Result;
use native_core_infra::checkpoint::{Dtype, Manifest};
use native_core_infra::model_load::{self, Loaded, Options, Pending, Request, Verify};
use std::ffi::{c_int, c_void};
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

extern "C" {
    fn w21_bicubic(src: *const c_void, in_h: i64, in_w: i64, iy: *const c_void, wy: *const c_void, ix: *const c_void,
                   wx: *const c_void, out: *mut c_void, out_h: i64, out_w: i64, channel_stride: i64, s: crate::cuda::Stream)
                   -> c_int;
}

const TEXT_LEN: usize = 512;
const TEXT_DIM: usize = 4096;
const CLIP_TOKENS: usize = 257;
const CLIP_DIM: usize = 1280;

fn load_options(declaration: &native_core_infra::checkpoint::Declaration) -> Result<Options> {
    let mut options = Options::for_declaration(declaration)?;
    // NativeService declares light verification and owns the post-request audit.
    // Preserve the one-shot CLI's full-check default when it has no audit owner.
    if std::env::var_os("NATIVE_LOAD_VERIFY").is_none() { options.verify = Verify::Full; }
    Ok(options)
}

/// The four models of the i2v-14B pipeline.
pub struct Models<'d> {
    pub(crate) startup: Option<crate::startup::WanStartup>,
    pub(crate) parts: Rc<std::cell::RefCell<crate::startup::Parts<'d>>>,
    pub device: &'d Device,
    /// The streaming loads' allocations the models' weights view (prepared checkpoints).
    pub weights: Vec<Rc<Loaded>>,
    /// Seconds each model took to build (startup diagnostics).
    pub setup: Vec<(&'static str, f64)>,
    /// With two GPUs, GPU 1's receipt for its copy of the DiT weights.
    pub(crate) replica: Option<serde_json::Value>,
    /// `generate_request` keeps the last request's conditioning when set (a serving owner without
    /// gold observers): the next request with the same `conditioning_key` reuses it.
    pub reuse_conditioning: bool,
    conditioning: Option<(String, Conditioning)>,
}
/// What `condition` reads from a request: a request with the same key has the same T5, CLIP and
/// VAE conditioning (deterministic kernels on the same weights).
pub fn conditioning_key(prepared: &Prepared, salt: &str) -> String {
    fn bytes<T: Copy>(v: &[T]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) } }
    let mut key = salt.as_bytes().to_vec();
    for (ids, mask, len) in &prepared.tokens {
        key.extend_from_slice(bytes(ids)); key.extend_from_slice(bytes(mask)); key.extend_from_slice(&len.to_le_bytes());
    }
    for v in [&prepared.image, &prepared.to_tensor, &prepared.taps_y.1, &prepared.taps_x.1] { key.extend_from_slice(bytes(v)); }
    for v in [&prepared.taps_y.0, &prepared.taps_x.0] { key.extend_from_slice(bytes(v)); }
    for n in [prepared.src_h, prepared.src_w, prepared.lat_h, prepared.lat_w, prepared.request.frame_num] {
        key.extend_from_slice(&n.to_le_bytes());
    }
    crate::gold::sha256_hex(&key)
}

/// The tensors i2v-14B uses from a prepared checkpoint manifest (core_infra/runtime/checkpoint.py,
/// models/wan2.1/prepare.py), each in the dtype its model runs it in: the DiT's
/// bf16 copy and float64 `freqs`, T5 bf16, VAE fp32 as published, and the CLIP visual tower fp16,
/// converted on the device from the published fp32 as the reference's load_state_dict into fp16
/// parameters rounds it.
pub fn prepared_requests(manifest: &Manifest) -> Vec<Request> {
    manifest.tensors.iter().filter_map(|(name, t)| {
        let dtype = if name.starts_with("dit.") || name.starts_with("t5.") || name.starts_with("vae.") { t.dtype }
            else if name.starts_with("clip.visual.") { if t.dtype == Dtype::F32 { Dtype::F16 } else { t.dtype } }
            else { return None };
        Some(Request { name: name.clone(), dtype })
    }).collect()
}

/// Starts the streaming load of a prepared checkpoint: readers only, so call it before
/// `Device::open` and file reads overlap CUDA context creation. `prepared` holds manifest.json
/// (admitted through NATIVE_CHECKPOINT_* or the required offline WAN21_MANIFEST_SHA256 pin);
/// `checkpoint` is the published checkpoint directory.
pub fn start_prepared(prepared: &Path, checkpoint: &Path) -> Result<Pending> {
    start_prepared_where(prepared, checkpoint, |_| true)
}

/// `start_prepared` for the tensors whose manifest names `keep` accepts (e.g. the encoders first,
/// the DiT in a second load that streams while they run).
pub fn start_prepared_where(prepared: &Path, checkpoint: &Path, keep: impl Fn(&str) -> bool) -> Result<Pending> {
    let declaration = crate::admission::checkpoint(prepared, checkpoint)?;
    let options = load_options(&declaration)?;
    let requests: Vec<Request> = prepared_requests(&declaration.manifest).into_iter().filter(|r| keep(&r.name)).collect();
    Ok(model_load::start(&declaration.manifest, &declaration.roots, &requests, options)?)
}

impl<'d> Models<'d> {
    pub fn startup_report(&self)->serde_json::Value {
        let mut report=self.startup.as_ref().map(|startup|startup.report()).unwrap_or_else(||serde_json::json!({"diagnostic_converted_weights":true}));
        if let Some(replica)=&self.replica {report["gpu1_dit_replica"]=replica.clone();}
        report
    }
    /// The GPUs the request runs on: 2 when GPU 1 runs the unconditional pass.
    pub fn world(&self)->usize {if self.parts.borrow().peer.is_some() {2} else {1}}
    /// `weights`: t5/clip/vae/dit safetensors. The DiT's buffers are sized for `grid` (re-sized per
    /// request by `generate`).
    pub fn load(weights: &Path, grid: Grid, device: &'d Device) -> Result<Self> {
        Ok(Self {
            startup:None,
            parts:Rc::new(std::cell::RefCell::new(crate::startup::Parts {
                t5: Some(T5::load(&SafeTensors::open(weights.join("t5.safetensors"))?, device)?),
                clip: Some(ClipVisual::load(&SafeTensors::open(weights.join("clip.safetensors"))?, device)?),
                vae: Some(Vae::load(&SafeTensors::open(weights.join("vae.safetensors"))?, device)?),
                dit: Some(Dit::load(&weights.join("dit.safetensors"), grid, device)?),
                device:Some(device),weights:vec![],peer:None,replica:None})),
            device,
            weights: Vec::new(),
            setup: Vec::new(),
            replica: None,
            reuse_conditioning: false,
            conditioning: None,
        })
    }

    /// The models over a finished streaming load (`start_prepared` + `Pending::finish`).
    pub fn from_loaded(loaded: Loaded, grid: Grid, device: &'d Device) -> Result<Self> {
        let loaded = Rc::new(loaded);
        let part = |prefix: &str| DeviceWeights { loaded: loaded.clone(), prefix: prefix.to_string() };
        let mut setup = Vec::new();
        let mut timed = |name: &'static str, t: Instant| -> Result<()> { device.sync()?; setup.push((name, t.elapsed().as_secs_f64())); Ok(()) };
        let t = Instant::now();
        let t5 = T5::load(&part("t5."), device)?;
        timed("t5", t)?;
        let t = Instant::now();
        let clip = ClipVisual::load(&part("clip."), device)?;
        timed("clip", t)?;
        let t = Instant::now();
        let vae = Vae::load(&part("vae."), device)?;
        timed("vae", t)?;
        let t = Instant::now();
        let dit = Dit::load_from(&part("dit."), grid, device)?;
        timed("dit", t)?;
        Ok(Self {startup:None,parts:Rc::new(std::cell::RefCell::new(crate::startup::Parts {
            t5:Some(t5),clip:Some(clip),vae:Some(vae),dit:Some(dit),device:Some(device),weights:vec![loaded.clone()],
            peer:None,replica:None})),device,weights:vec![loaded],setup,replica:None,reuse_conditioning:false,conditioning:None})
    }
}

/// Builds the request's cuDNN attention plans (DiT for its grid, CLIP) ahead of model setup, e.g.
/// on a thread that runs while the weights stream in.
pub fn prewarm(prepared: &Prepared) -> Result<()> {
    ClipVisual::prewarm()?;
    Dit::prewarm(grid(prepared))
}

/// Wall-clock stamps (UNIX seconds) of a request's phases, in order.
pub type Marks = std::cell::RefCell<Vec<(String, f64)>>;

pub fn unix_now() -> f64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64() }

/// The first request of a cold process through core's Component/Preparation/
/// DeclaredStartup executor. Wan supplies its original tensor groups and math;
/// core owns readers, CUDA release, preparations, stage progression and joins.
/// Returns the device and the resident models for later requests
/// with the result; `observe` sees the gold-named tensors, and returning an error from it stops the
/// request (its error is returned with the models dropped). `marks` records each phase.
pub fn cold_start(prepared_dir: &Path, checkpoint_dir: &Path, front: impl FnOnce() -> std::result::Result<Prepared, String> + Send + 'static,
                  marks: &Marks, observe: &mut dyn FnMut(&str, u64) -> Result<()>)
                  -> Result<(&'static Device, Models<'static>, Prepared, Generated)> {
    cold_start_on(1, prepared_dir, checkpoint_dir, front, marks, observe)
}

/// `cold_start` on `world` GPUs (1 or 2). With two, GPU 1 runs every unconditional DiT pass
/// (cfg_parallel.rs). It opens while GPU 0 loads, builds the request's attention plans, and copies
/// GPU 0's DiT weights over the peer link, so the checkpoint is still read once.
pub fn cold_start_on(world: usize, prepared_dir: &Path, checkpoint_dir: &Path,
                     front: impl FnOnce() -> std::result::Result<Prepared, String> + Send + 'static,
                     marks: &Marks, observe: &mut dyn FnMut(&str, u64) -> Result<()>)
                     -> Result<(&'static Device, Models<'static>, Prepared, Generated)> {
    let _first_dit=native_core_infra::execution_start::Capture::begin();
    if !matches!(world, 1 | 2) { return Err("official Wan runs on one or two GPUs".into()); }
    if world == 2 { crate::dit::plan_attention_split(); }
    crate::admission::require_backend_preparation()?;
    // Reject absent or changed metadata before starting readers, CPU work or CUDA.
    let declaration = crate::admission::checkpoint(prepared_dir, checkpoint_dir)?;
    let options = load_options(&declaration)?;
    let peer = if world == 2 { Some(crate::cfg_parallel::CfgWorker::start()?) } else { None };
    let mut startup=crate::startup::WanStartup::new(declaration,options)?;
    startup.parts.borrow_mut().peer=peer;
    startup.start()?;startup.request(front)?;
    native_core_infra::startup_trace::measure("wan_phase","wait for VAE readiness",0,||startup.ready(VIDEO))?;
    let parts=startup.parts.clone();let device=parts.borrow().device.ok_or("Wan device missing")?;
    let prepared=startup.prepared()?;
    let prewarm=parts.borrow().peer.as_ref().map(|peer|peer.begin_prewarm(grid(&prepared))).transpose()?;
    let t_start = Instant::now();
    let mut timings = Timings::default();
    let cond = native_core_infra::startup_trace::measure("wan_phase","conditioning through existing sync",0,||->Result<_>{
        let y=native_core_infra::startup_trace::measure("wan_phase","video conditioning",0,||startup.invoke(VIDEO,||
            condition_video(parts.borrow().vae.as_ref().ok_or("VAE preparation missing")?,device,&prepared,observe)))?;
        native_core_infra::startup_trace::measure("wan_phase","wait for encoders readiness",0,||startup.ready(ENCODERS))?;
        let clip_fea=native_core_infra::startup_trace::measure("wan_phase","image conditioning",0,||startup.invoke(ENCODERS,||
            condition_clip(parts.borrow().clip.as_ref().ok_or("CLIP preparation missing")?,device,&prepared,observe)))?;
        let contexts=native_core_infra::startup_trace::measure("wan_phase","text conditioning",0,||startup.invoke(ENCODERS,|| {
            condition_text(parts.borrow().t5.as_ref().ok_or("T5 preparation missing")?,device,&prepared,observe)
        }))?;
        device.sync()?;Ok(Conditioning{contexts,clip_fea,y})
    })?;
    timings.condition = t_start.elapsed().as_secs_f64();
    native_core_infra::startup_trace::measure("wan_phase","wait for DiT readiness",0,||startup.ready("dit"))?;
    if let Some(prewarm)=prewarm {prewarm.finish()?;}
    let replica=parts.borrow_mut().replica.take();
    let replica=replica.map(|replica|native_core_infra::startup_trace::measure("wan_phase","wait for the GPU 1 DiT",0,||replica.finish())).transpose()?;
    let t_sampling = Instant::now();
    let latent = startup.invoke("dit",||denoise_peer(&parts,device,&prepared,&cond,observe))?;
    // Kept for a serving owner that enables reuse_conditioning (dropped with the models otherwise).
    let kept=Some((conditioning_key(&prepared,""),cond));
    device.sync()?;
    timings.sampling = t_sampling.elapsed().as_secs_f64();
    let t_decode = Instant::now();
    let (video, frames) = startup.invoke(VIDEO,||decode(parts.borrow().vae.as_ref().ok_or("VAE preparation missing")?,&prepared,&latent,observe))?;
    device.sync()?;
    timings.vae_decode = t_decode.elapsed().as_secs_f64();
    timings.total = t_start.elapsed().as_secs_f64();
    drop(prepared);let prepared=startup.take_prepared()?;
    marks.borrow_mut().extend(startup.marks());
    let weights=parts.borrow().weights.clone();
    let models=Models {startup:Some(startup),parts,device,weights,setup:vec![],replica,reuse_conditioning:false,conditioning:kept};
    Ok((device, models, prepared, Generated { video, frames, timings }))
}

/// The DiT grid of a prepared request.
pub fn grid(prepared: &Prepared) -> Grid {
    Grid { frames: prepared.latent_frames(), height: prepared.lat_h, width: prepared.lat_w }
}

/// Wall time of each stage (device-synchronized), in seconds.
#[derive(Debug, Default, serde::Serialize)]
pub struct Timings {
    /// T5, CLIP and the VAE encode (`condition`).
    pub condition: f64,
    pub sampling: f64,
    pub vae_decode: f64,
    pub total: f64,
}

pub struct Generated {
    /// fp32 [3, frames, 8 lat_h, 8 lat_w] in [-1, 1] (generate's return value).
    pub video: Buffer,
    pub frames: usize,
    pub timings: Timings,
}

/// F.interpolate(img[None], (h, w), mode='bicubic') of the fp32 [3, src_h, src_w] image into frame 0
/// of the contiguous fp32 [3, frames, h, w] video, with frontend::cubic_taps (indices i32, weights f32).
#[allow(clippy::too_many_arguments)]
pub fn bicubic(image: u64, src_h: usize, src_w: usize, iy: u64, wy: u64, ix: u64, wx: u64, video: u64, h: usize, w: usize,
               frames: usize, device: &Device) -> Result<()> {
    let p = |a: u64| a as *const c_void;
    check(unsafe { w21_bicubic(p(image), src_h as i64, src_w as i64, p(iy), p(wy), p(ix), p(wx), video as *mut c_void, h as i64,
                               w as i64, (frames * h * w) as i64, device.stream) }, "bicubic")
}

fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

/// The request's conditioning: the T5 contexts (prompt, negative prompt) with their token counts,
/// the CLIP image features and y = cat([first-frame mask, VAE latent]).
pub struct Conditioning { pub contexts: Vec<(Buffer, usize)>, pub clip_fea: Buffer, pub y: Buffer }

/// generate's preamble: T5 on both prompts, CLIP on the image, and the VAE encode of the
/// conditioning video ([bicubic(img), 0, ..., 0]) with the first-frame mask. Needs no DiT, so it
/// can run while the DiT's weights are still streaming in.
pub fn condition(t5: &T5, clip: &ClipVisual, vae: &Vae, device: &Device, prepared: &Prepared,
                 observe: &mut dyn FnMut(&str, u64) -> Result<()>) -> Result<Conditioning> {
    let contexts = condition_text(t5, device, prepared, observe)?;
    let (clip_fea, y) = condition_image(clip, vae, device, prepared, observe)?;
    Ok(Conditioning { contexts, clip_fea, y })
}

/// A traced conditioning step. With WAN_PHASE_SYNC=1 the span also waits for the step's GPU work
/// (a diagnostic that adds a stream synchronization); by default it times the host side only.
fn step<T>(name: &str, device: &Device, work: impl FnOnce() -> Result<T>) -> Result<T> {
    native_core_infra::startup_trace::measure("wan_step", name, 0, || {
        let value = work()?;
        if std::env::var_os("WAN_PHASE_SYNC").is_some() { device.sync()?; }
        Ok(value)
    })
}

/// The T5 contexts of the prompt and the negative prompt, with their token counts.
pub fn condition_text(t5: &T5, device: &Device, prepared: &Prepared,
                      observe: &mut dyn FnMut(&str, u64) -> Result<()>) -> Result<Vec<(Buffer, usize)>> {
    let mut contexts = Vec::new();
    for (call, (ids, mask, len)) in prepared.tokens.iter().enumerate() {
        let (ids, mask) = (Buffer::upload(bytes_of(ids), device)?, Buffer::upload(bytes_of(mask), device)?);
        observe(&format!("t5/call{call}/ids"), ids.dev())?;
        observe(&format!("t5/call{call}/mask"), mask.dev())?;
        let out = Buffer::new(TEXT_LEN * TEXT_DIM * 2)?;
        step("t5 forward", device, || t5.forward(ids.dev(), mask.dev(), 1, out.dev()))?;
        observe(&format!("t5/call{call}/context"), out.dev())?;
        contexts.push((out, *len));
    }
    Ok(contexts)
}

/// The CLIP image features and y = cat([first-frame mask, VAE latent]).
pub fn condition_image(clip: &ClipVisual, vae: &Vae, device: &Device, prepared: &Prepared,
                       observe: &mut dyn FnMut(&str, u64) -> Result<()>) -> Result<(Buffer, Buffer)> {
    let clip_fea = condition_clip(clip, device, prepared, observe)?;
    let y = condition_video(vae, device, prepared, observe)?;
    Ok((clip_fea, y))
}

/// The CLIP image features of the request image.
pub fn condition_clip(clip: &ClipVisual, device: &Device, prepared: &Prepared,
                      observe: &mut dyn FnMut(&str, u64) -> Result<()>) -> Result<Buffer> {
    let image = Buffer::upload(bytes_of(&prepared.image), device)?;
    observe("clip/in/video", image.dev())?;
    let clip_fea = Buffer::new(CLIP_TOKENS * CLIP_DIM * 2)?;
    step("clip forward", device, || clip.forward(image.dev(), prepared.src_h, prepared.src_w, clip_fea.dev()))?;
    observe("clip/out", clip_fea.dev())?;
    Ok(clip_fea)
}

/// y = cat([first-frame mask, VAE latent of [bicubic(img), 0, ..., 0]]). Needs only the VAE, the
/// smallest weights, so cold startup runs the heaviest conditioning step first.
pub fn condition_video(vae: &Vae, device: &Device, prepared: &Prepared,
                       observe: &mut dyn FnMut(&str, u64) -> Result<()>) -> Result<Buffer> {
    let r = &prepared.request;
    let image = Buffer::upload(bytes_of(&prepared.image), device)?;
    let (frames, h, w) = (r.frame_num, prepared.height(), prepared.width());
    let video = VTensor::new(3, frames, h, w)?;
    video.buf.zero(device)?;
    let upload = |v: &[u8]| Buffer::upload(v, device);
    let (iy, wy) = (upload(bytes_of(&prepared.taps_y.0))?, upload(bytes_of(&prepared.taps_y.1))?);
    let (ix, wx) = (upload(bytes_of(&prepared.taps_x.0))?, upload(bytes_of(&prepared.taps_x.1))?);
    bicubic(image.dev(), prepared.src_h, prepared.src_w, iy.dev(), wy.dev(), ix.dev(), wx.dev(), video.dev(), h, w, frames, device)?;
    observe("vae/encode/in", video.dev())?;
    let latent_y = step("vae encode", device, || vae.encode(&video))?;
    drop(video);
    observe("vae/encode/out", latent_y.dev())?;
    let g = grid(prepared);
    let plane = g.frames * g.height * g.width;
    // y = cat([msk, vae latent]): msk is 1 on latent frame 0 of its 4 channels, 0 elsewhere.
    let y = Buffer::new((4 + Z_DIM) * plane * 4)?;
    let mut mask = vec![0f32; 4 * plane];
    for c in 0..4 { mask[c * plane..c * plane + g.height * g.width].fill(1.0); }
    upload_into(&y, bytes_of(&mask), device)?;
    check(unsafe { memcpy_d2d(y.at(4 * plane * 4), latent_y.dev(), Z_DIM * plane * 4, device.stream) }, "y concat")?;
    Ok(y)
}

/// The sampling loop: the seeded noise, then `sample_steps` UniPC steps of two DiT passes and
/// classifier-free guidance. Returns the final latent (fp32 [16, F, h, w]).
pub fn denoise(dit: &mut Dit, device: &Device, prepared: &Prepared, cond: &Conditioning,
               observe: &mut dyn FnMut(&str, u64) -> Result<()>) -> Result<Buffer> {
    denoise_with_peer(dit,device,prepared,cond,observe,None)
}
pub(crate) fn denoise_with_peer(dit:&mut Dit,device:&Device,prepared:&Prepared,cond:&Conditioning,
    observe:&mut dyn FnMut(&str,u64)->Result<()>,peer:Option<&crate::cfg_parallel::CfgWorker>)->Result<Buffer>{
    let (st, r, g) = (device.stream, &prepared.request, grid(prepared));
    let Conditioning { contexts, clip_fea, y } = cond;
    dit.set_grid(g)?;
    let n = Z_DIM * g.frames * g.height * g.width;
    let latent = Buffer::new(n * 4)?;
    CudaGenerator::new(r.seed)?.randn(latent.dev(), n, false, st)?;
    observe("noise/randn0", latent.dev())?;
    let ctx = [dit.pass_context(contexts[0].0.dev(), contexts[0].1, clip_fea.dev())?,
               dit.pass_context(contexts[1].0.dev(), contexts[1].1, clip_fea.dev())?];
    // One GPU: the unconditional pass starts from the conditional pass's block-0 prefix. Two GPUs:
    // the passes split that prefix by token rows instead, and GPU 1, done with its pass first,
    // computes half the rows of the conditional pass's last block.
    let prefix=if peer.is_none(){Some(crate::dit::Prefix::new(g)?)}else{None};
    let mut unipc = UniPc::new(r.sample_steps, r.shift, n)?;
    let timesteps = unipc.timesteps.clone();
    if let Some(peer)=peer{
        let(sent,meet,first)=(peer.split_events()?,std::sync::Arc::new(crate::dit::Rendezvous::default()),g.tokens()/2);
        let [k,v,_,_]=Dit::context_kv(&ctx[0],crate::dit::LAYERS-1);
        let split=crate::cfg_parallel::Split{bufs:dit.split_buffers(),sent,meet:meet.clone(),first,last_text:[k,v]};
        let(peer_bufs,peer_sent)=peer.configure(g,&contexts[1].0,contexts[1].1,clip_fea,y,device,Some(split),&timesteps)?;
        dit.set_split(Some(crate::dit::AttnSplit{rows:(0,first),device:0,peer_device:1,peer:peer_bufs,sent,peer_sent,meet,
            main:true,last_text:None}))?;
    }
    let share=|pass:usize|match(&prefix,pass){(Some(prefix),0)=>Share::Store(prefix),(Some(prefix),_)=>Share::Reuse(prefix),
        (None,_)=>Share::None};
    let (out_c, out_u, guided) = (Buffer::new(n * 4)?, Buffer::new(n * 4)?, Buffer::new(n * 4)?);
    let (mut cur, mut next) = (latent, Buffer::new(n * 4)?);
    // Every step's timestep embeddings ahead of the loop, and the timesteps on the device in one upload,
    // so that no step waits for the host.
    dit.prepare_times(&timesteps)?;
    let t_dev = Buffer::new(8 * timesteps.len().max(1))?;
    let bytes: Vec<u8> = timesteps.iter().flat_map(|t| t.to_le_bytes()).collect();
    upload_into(&t_dev, &bytes, device)?;
    let executed=native_core_infra::switch_benchmark::steps(timesteps.len(),false)?;
    for (step, &t) in timesteps.iter().take(executed).enumerate() {
        // GPU 1 starts once the latent is ready; the timestep embeddings only need t.
        let mut secondary=peer.map(|peer|peer.forward(cur.dev(),t,out_u.dev(),device)).transpose()?;
        dit.time(t)?;
        let t_step = t_dev.dev() + 8 * step as u64;
        for (pass, out) in [&out_c, &out_u].into_iter().enumerate() {
            let f = 2 * step + pass;
            observe(&format!("dit/f{f:04}/in/x"), cur.dev())?;
            observe(&format!("dit/f{f:04}/in/t"), t_step)?;
            observe(&format!("dit/f{f:04}/in/context"), contexts[pass].0.dev())?;
            observe(&format!("dit/f{f:04}/in/clip_fea"), clip_fea.dev())?;
            observe(&format!("dit/f{f:04}/in/y"), y.dev())?;
            if pass==1 && peer.is_some(){secondary.take().ok_or("Wan CFG forward missing")?.finish()?;}
            else{dit.forward_with(cur.dev(), y.dev(), &ctx[pass], out.dev(), &mut |_, _| Ok(()), share(pass))?;}
            observe(&format!("dit/f{f:04}/out"), out.dev())?;
        }
        unipc.guidance(out_c.dev(), out_u.dev(), guided.dev(), r.guide_scale, st)?;
        let name = |s: &str| format!("sched/step{step:02}/{s}");
        observe(&name("model_output"), guided.dev())?;
        observe(&name("sample"), cur.dev())?;
        unipc.step(guided.dev(), cur.dev(), next.dev(), device, &mut |what, addr| observe(&name(what), addr))?;
        observe(&name("prev_sample"), next.dev())?;
        std::mem::swap(&mut cur, &mut next);
    }
    Ok(cur)
}

/// `denoise` with the declared startup's DiT and, on two GPUs, its peer.
fn denoise_peer(parts:&std::cell::RefCell<crate::startup::Parts>,device:&Device,prepared:&Prepared,cond:&Conditioning,
                observe:&mut dyn FnMut(&str,u64)->Result<()>)->Result<Buffer> {
    let mut parts=parts.borrow_mut();let parts=&mut *parts;
    denoise_with_peer(parts.dit.as_mut().ok_or("DiT preparation missing")?,device,prepared,cond,observe,parts.peer.as_ref())
}

/// The VAE decode of the final latent: fp32 video [3, frames, 8 lat_h, 8 lat_w] in [-1, 1].
pub fn decode(vae: &Vae, prepared: &Prepared, latent: &Buffer, observe: &mut dyn FnMut(&str, u64) -> Result<()>)
              -> Result<(Buffer, usize)> {
    decode_with_start(vae,prepared,latent,observe,|_|Ok(()))
}
pub fn decode_with_start(vae:&Vae,prepared:&Prepared,latent:&Buffer,observe:&mut dyn FnMut(&str,u64)->Result<()>,
    start:impl FnOnce(crate::cuda::Stream)->Result<()>)->Result<(Buffer,usize)>{
    let g = grid(prepared);
    observe("vae/decode/in", latent.dev())?;
    let (video, frames) = vae.decode_with_start(latent.dev(), g.frames, g.height, g.width,start)?;
    observe("vae/decode/out", video.dev())?;
    observe("video/out", video.dev())?;
    Ok((video, frames))
}

/// generate(prompt, img, ...) for a prepared request: condition, denoise, decode. `observe(name,
/// address)` sees every boundary tensor under the gold's names (an address holds at least the
/// gold tensor's bytes).
pub fn generate(models: &mut Models, prepared: &Prepared, observe: &mut dyn FnMut(&str, u64) -> Result<()>)
                -> Result<Generated> {
    if models.startup.is_some() {
        let prepared=prepared.clone();
        return generate_request(models,move||Ok(prepared),observe).map(|(_,generated)|generated);
    }
    let _first_dit=native_core_infra::execution_start::Capture::begin();
    let device = models.device;
    let t_start = Instant::now();
    let mut timings = Timings::default();
    let mut since = Instant::now();
    let mut lap = |slot: &mut f64| -> Result<()> {
        device.sync()?;
        *slot = since.elapsed().as_secs_f64();
        since = Instant::now();
        Ok(())
    };
    let mut parts=models.parts.borrow_mut();
    let cond = condition(parts.t5.as_ref().ok_or("T5 missing")?,parts.clip.as_ref().ok_or("CLIP missing")?,parts.vae.as_ref().ok_or("VAE missing")?, device, prepared, observe)?;
    lap(&mut timings.condition)?;
    let latent = denoise(parts.dit.as_mut().ok_or("DiT missing")?, device, prepared, &cond, observe)?;
    drop(cond);
    lap(&mut timings.sampling)?;
    let (video, frames) = decode(parts.vae.as_ref().ok_or("VAE missing")?, prepared, &latent, observe)?;
    lap(&mut timings.vae_decode)?;
    timings.total = t_start.elapsed().as_secs_f64();
    Ok(Generated { video, frames, timings })
}

/// A resident request uses the same declaration as cold startup, including CPU
/// preprocessing, backend preparation and the gates around actual computation.
/// Large frontend tensors move through core's typed slot without a clone.
pub fn generate_request(models:&mut Models,front:impl FnOnce()->std::result::Result<Prepared,String>+Send+'static,
                        observe:&mut dyn FnMut(&str,u64)->Result<()>)->Result<(Prepared,Generated)> {
    let _first_dit=native_core_infra::execution_start::Capture::begin();
    let parts=models.parts.clone();let device=models.device;
    let startup=models.startup.as_mut().ok_or("resident request requires declared startup")?;
    startup.request(front)?;startup.ready(VIDEO)?;
    let prepared=startup.prepared()?;
    let t_start=Instant::now();let mut timings=Timings::default();let mut since=Instant::now();
    let mut lap=|slot:&mut f64|->Result<()> {device.sync()?;*slot=since.elapsed().as_secs_f64();since=Instant::now();Ok(())};
    let key=models.reuse_conditioning.then(||conditioning_key(&prepared,""));
    let cached=match (key.as_ref(),models.conditioning.take()){(Some(key),Some((k,c))) if *key==k=>Some(c),_=>None};
    let cond=if let Some(cond)=cached{startup.ready(ENCODERS)?;cond}else{
        let y=startup.invoke(VIDEO,||condition_video(parts.borrow().vae.as_ref().ok_or("VAE preparation missing")?,device,&prepared,observe))?;
        startup.ready(ENCODERS)?;
        let clip_fea=startup.invoke(ENCODERS,||condition_clip(parts.borrow().clip.as_ref().ok_or("CLIP preparation missing")?,device,&prepared,observe))?;
        let contexts=startup.invoke(ENCODERS,||condition_text(parts.borrow().t5.as_ref().ok_or("T5 preparation missing")?,device,&prepared,observe))?;
        Conditioning{contexts,clip_fea,y}
    };
    lap(&mut timings.condition)?;
    let latent=startup.invoke("dit",||denoise_peer(&parts,device,&prepared,&cond,observe))?;
    if let Some(key)=key{models.conditioning=Some((key,cond));}else{drop(cond);}
    lap(&mut timings.sampling)?;
    let (video,frames)=startup.invoke(VIDEO,||decode(parts.borrow().vae.as_ref().ok_or("VAE preparation missing")?,&prepared,&latent,observe))?;
    lap(&mut timings.vae_decode)?;timings.total=t_start.elapsed().as_secs_f64();
    drop(prepared);let prepared=startup.take_prepared()?;
    Ok((prepared,Generated {video,frames,timings}))
}

/// Writes the video as an MP4 through `ffmpeg` (libx264, CRF 10 as imageio's quality=8, 16 fps as
/// generate.py's cache_video). Fails early when ffmpeg is missing.
pub fn write_mp4(generated: &Generated, prepared: &Prepared, device: &Device, path: &Path) -> Result<()> {
    write_mp4_after_download(generated,prepared,device,path,||Ok(()))
}
/// Publish the CPU-only encoding tail after the final GPU readback.
pub fn write_mp4_after_download(generated:&Generated,prepared:&Prepared,device:&Device,
    path:&Path,gpu_complete:impl FnOnce()->Result<()>)->Result<()> {
    let video=generated.video.download_f32(device)?;
    gpu_complete()?;
    write_mp4_host(&video,generated.frames,prepared,path)
}
/// Encode the same completed host frames used by output verification.
pub fn write_mp4_host(video:&[f32],frame_count:usize,prepared:&Prepared,path:&Path)->Result<()> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let (h, w) = (prepared.height(), prepared.width());
    if frame_count.checked_mul(h).and_then(|n|n.checked_mul(w)).and_then(|n|n.checked_mul(3))!=Some(video.len()) {
        return Err("host video frame extent differs".into());
    }
    let frames = video_to_u8(video, frame_count, h, w);
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) { std::fs::create_dir_all(dir)?; }
    let mut ffmpeg = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", "16", "-i", "-",
               "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "10"])
        .arg(path).stdin(Stdio::piped()).spawn().map_err(|e| format!("ffmpeg: {e}"))?;
    ffmpeg.stdin.take().ok_or("ffmpeg stdin")?.write_all(&frames)?;
    if !ffmpeg.wait()?.success() { return Err("ffmpeg failed".into()); }
    Ok(())
}

pub use crate::frontend::video_to_u8;
