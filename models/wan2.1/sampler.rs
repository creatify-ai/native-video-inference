//! FlowUniPCMultistepScheduler (wan/utils/fm_solvers_unipc.py: order 2, bh2, predict_x0,
//! lower_order_final, final sigma 0) and the classifier-free guidance of WanI2V.generate, with the
//! reference's numerics: the schedule in numpy float64, per-step scalars as fp32 0-dim CPU tensors
//! (MKL VML log / expm1 from schedules.rs, true division, MKL's 2x2 solve), tensor updates as CPU
//! fp32 eager ops (kernels/sched.cu).
use crate::cuda::{check, Buffer, Device, Stream};
use crate::Result;
use std::ffi::{c_int, c_void};

extern "C" {
    fn w21_cfg(cond: *const c_void, uncond: *const c_void, out: *mut c_void, guide: f32, n: i64, s: Stream) -> c_int;
    fn w21_convert(sample: *const c_void, mo: *const c_void, x0: *mut c_void, sigma: f32, n: i64, s: Stream) -> c_int;
    fn w21_unipc_update(x: *const c_void, m0: *const c_void, m1: *const c_void, mt: *const c_void, out: *mut c_void, s1: f32,
                        s2: f32, s3: f32, rk: f32, rho0: f32, rho1: f32, corrector: c_int, order: c_int, n: i64, s: Stream) -> c_int;
}
fn p(a: u64) -> *mut c_void { a as *mut c_void }

pub const NUM_TRAIN_TIMESTEPS: usize = 1000;
const SOLVER_ORDER: usize = 2;

/// numpy.linspace(start, stop, num): i * step + start, the last element exactly stop.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let step = (stop - start) / (num - 1) as f64;
    (0..num).map(|i| if i == num - 1 { stop } else { i as f64 * step + start }).collect()
}

/// set_timesteps(steps, shift) after __init__(num_train_timesteps=1000, shift=1): (fp32 sigmas with
/// the final 0, int64 timesteps the DiT sees).
pub fn schedule(steps: usize, shift: f64) -> (Vec<f32>, Vec<i64>) {
    // __init__: alphas = linspace(1, 1/1000, 1000)[::-1]; sigmas = fp32(1 - alphas) (shift 1 is exact).
    let alphas: Vec<f64> = linspace(1.0, 1.0 / NUM_TRAIN_TIMESTEPS as f64, NUM_TRAIN_TIMESTEPS).into_iter().rev().collect();
    let sigma_max = (1.0 - alphas[0]) as f32 as f64;
    let sigma_min = (1.0 - alphas[NUM_TRAIN_TIMESTEPS - 1]) as f32 as f64;
    let mut sig: Vec<f64> = linspace(sigma_max, sigma_min, steps + 1);
    sig.truncate(steps);
    let sig: Vec<f64> = sig.iter().map(|&s| shift * s / (1.0 + (shift - 1.0) * s)).collect();
    let timesteps = sig.iter().map(|&s| (s * NUM_TRAIN_TIMESTEPS as f64) as i64).collect();
    let mut sigmas: Vec<f32> = sig.iter().map(|&s| s as f32).collect();
    sigmas.push(0.0);
    (sigmas, timesteps)
}

/// The order-`order` scalars shared by predictor and corrector (bh2, predict_x0): ([s1, s2, s3, rk], b),
/// with lambda(sigma) = log(1 - sigma) - log(sigma) and expm1(-h) from the MKL tables (schedules.rs).
/// `t` is always `s0 + 1`.
fn coefficients(sigmas: &[f32], lambda: &[f32], expm1: &[f32], t: usize, s0: usize, si: Option<usize>, order: usize)
                -> ([f32; 4], Vec<f32>) {
    let (sigma_t, sigma_s0) = (sigmas[t], sigmas[s0]);
    let alpha_t = 1.0f32 - sigma_t;
    let h = lambda[t] - lambda[s0];
    let rk = si.map_or(1.0, |si| (lambda[si] - lambda[s0]) / h);
    let hh = -h;
    let h_phi_1 = expm1[s0];
    let mut h_phi_k = h_phi_1 / hh - 1.0;
    let b_h = expm1[s0];
    let mut factorial = 1u32;
    let mut b = Vec::new();
    for i in 1..=order {
        b.push((h_phi_k * factorial as f32) / b_h);
        factorial *= i as u32 + 1;
        h_phi_k = h_phi_k / hh - (1.0f64 / factorial as f64) as f32;
    }
    ([sigma_t / sigma_s0, alpha_t * h_phi_1, alpha_t * b_h, rk], b)
}

/// torch.linalg.solve(R, b) for R = [[1, 1], [rk, 1]] as MKL's sgesv computes it under
/// MKL_CBWR=COMPATIBLE (the reference's setting): partial pivoting, the multiplier as a reciprocal
/// product, no FMA, back substitution by multiplying with the diagonal's reciprocal. Fitted
/// bit-exactly on every solve of the 40-step schedule.
fn solve2(rk: f32, b: &[f32]) -> [f32; 2] {
    let (mut a, mut rhs) = ([[1.0f32, 1.0], [rk, 1.0]], [b[0], b[1]]);
    if rk.abs() > 1.0 {
        a.swap(0, 1);
        rhs.swap(0, 1);
    }
    let l = a[1][0] * (1.0 / a[0][0]);
    let u11 = a[1][1] - l * a[0][1];
    let y1 = rhs[1] - l * rhs[0];
    let x1 = y1 * (1.0 / u11);
    let x0 = (rhs[0] - a[0][1] * x1) * (1.0 / a[0][0]);
    [x0, x1]
}

/// The scheduler state of one request.
pub struct UniPc {
    pub sigmas: Vec<f32>,
    lambda: Vec<f32>,
    expm1: Vec<f32>,
    pub timesteps: Vec<i64>,
    model_outputs: [Option<Buffer>; SOLVER_ORDER],
    last_sample: Option<Buffer>,
    this_order: usize,
    lower_order_nums: usize,
    step_index: usize,
    n: usize,
}

impl UniPc {
    /// Errors for a schedule without MKL tables (schedules.rs lists the supported ones).
    pub fn new(steps: usize, shift: f64, elements: usize) -> Result<Self> {
        let (sigmas, timesteps) = schedule(steps, shift);
        let table = crate::schedules::TABLES.iter().find(|t| t.steps == steps && t.shift == shift)
            .ok_or_else(|| format!("no UniPC scalar table for {steps} steps, shift {shift} (see schedules.rs)"))?;
        let f = |v: &[u32]| v.iter().map(|&b| f32::from_bits(b)).collect::<Vec<f32>>();
        if f(table.sigmas) != sigmas { return Err("UniPC table sigmas differ from the computed schedule".into()); }
        Ok(Self { sigmas, lambda: f(table.lambda), expm1: f(table.expm1), timesteps, model_outputs: [None, None],
                  last_sample: None, this_order: 0, lower_order_nums: 0, step_index: 0, n: elements })
    }

    /// noise_pred = uncond + guide * (cond - uncond) into `out`.
    pub fn guidance(&self, cond: u64, uncond: u64, out: u64, guide: f64, st: Stream) -> Result<()> {
        check(unsafe { w21_cfg(p(cond), p(uncond), p(out), guide as f32, self.n as i64, st) }, "cfg")
    }

    /// One step(model_output, t, sample) -> prev_sample (all fp32 [n]); `observe` sees the
    /// converted output, the corrected sample and the predictor output under the gold's names.
    pub fn step(&mut self, model_output: u64, sample: u64, prev_sample: u64, device: &Device,
                observe: &mut dyn FnMut(&str, u64) -> Result<()>) -> Result<()> {
        let (st, n, i) = (device.stream, self.n, self.step_index);
        let converted = Buffer::new(n * 4)?;
        check(unsafe { w21_convert(p(sample), p(model_output), p(converted.dev()), self.sigmas[i], n as i64, st) }, "convert")?;
        observe("convert_model_output", converted.dev())?;
        let mut sample_now = sample;
        let corrected;
        if i > 0 && self.last_sample.is_some() {
            let order = self.this_order;
            let (c, b) = coefficients(&self.sigmas, &self.lambda, &self.expm1, i, i - 1, (order == 2).then(|| i - 2), order);
            let rhos = if order == 1 { [0.0, 0.5] } else { solve2(c[3], &b) };
            corrected = Buffer::new(n * 4)?;
            let m0 = self.model_outputs[SOLVER_ORDER - 1].as_ref().ok_or("corrector without a model output")?.dev();
            let m1 = self.model_outputs[SOLVER_ORDER - 2].as_ref().map_or(0, |b| b.dev());
            let x = self.last_sample.as_ref().unwrap().dev();
            check(unsafe { w21_unipc_update(p(x), p(m0), p(m1), p(converted.dev()), p(corrected.dev()), c[0], c[1], c[2], c[3],
                                            rhos[0], rhos[1], 1, order as c_int, n as i64, st) }, "unipc corrector")?;
            observe("multistep_uni_c_bh_update", corrected.dev())?;
            sample_now = corrected.dev();
        }
        let older = self.model_outputs[1].take();
        self.model_outputs[0] = older;
        self.model_outputs[1] = Some(converted);
        let this_order = SOLVER_ORDER.min(self.timesteps.len() - i);
        self.this_order = this_order.min(self.lower_order_nums + 1);
        let last = Buffer::new(n * 4)?;
        last.copy_from_device(sample_now, n * 4, device)?;
        self.last_sample = Some(last);
        let order = self.this_order;
        let (c, _) = coefficients(&self.sigmas, &self.lambda, &self.expm1, i + 1, i, (order == 2).then(|| i - 1), order);
        let m0 = self.model_outputs[1].as_ref().unwrap().dev();
        let m1 = self.model_outputs[0].as_ref().map_or(0, |b| b.dev());
        check(unsafe { w21_unipc_update(p(sample_now), p(m0), p(m1), p(0), p(prev_sample), c[0], c[1], c[2], c[3], 0.5, 0.0, 0,
                                        order as c_int, n as i64, st) }, "unipc predictor")?;
        observe("multistep_uni_p_bh_update", prev_sample)?;
        if self.lower_order_nums < SOLVER_ORDER { self.lower_order_nums += 1; }
        self.step_index += 1;
        Ok(())
    }
}
