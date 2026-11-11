//! torch-equivalent cuBLASLt GEMMs against the reference's own libcublasLt (kernels/cublaslt.cpp).
use crate::cuda::{Buffer, Stream};
use crate::Result;
use std::ffi::{c_char, c_void, CString};

extern "C" {
    fn dsv4_lt_init(path: *const c_char, err: *mut c_char, cap: usize) -> usize;
    fn dsv4_lt_linear(m: i32, n: i32, k: i32, x: u64, w: u64, out: u64, out_f32: i32, workspace: u64, cap: usize, sm_count: i32,
                      stream: *mut c_void, info: *mut i32, err: *mut c_char, cap2: usize) -> i32;
}

/// The reference's libcublasLt.so.13 (nvidia-cublas 13.1.1.3).
pub const LIB_SHA256: &str = "1489aeb75d55ee55782798343ac66f57771d561cc1fe1e0491568deefa0e6515";
/// torch's cuBLASLt workspace on this GPU (32 MiB).
pub const WORKSPACE: usize = 32 << 20;

/// `workspace` serves the main stream; streams added with `add_stream` get their own (as torch keeps
/// one cuBLAS workspace per stream), so GEMMs on concurrent streams never share one.
pub struct Lt { pub workspace: Buffer, pub sms: i32, streams: Vec<(usize, Buffer)> }

fn message(err: &[u8]) -> String {
    let n = err.iter().position(|&b| b == 0).unwrap_or(err.len());
    String::from_utf8_lossy(&err[..n]).into_owned()
}

impl Lt {
    /// Load the library at `path` after checking it is the reference's build.
    pub fn open(path: &str, sms: i32) -> Result<Self> {
        Self::verify(path)?;
        Self::init(path, sms)
    }

    /// Check that `path` is the reference's build (CPU only; also warms the page cache for dlopen).
    pub fn verify(path: &str) -> Result<()> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let sha = crate::gold::sha(&bytes);
        if sha != LIB_SHA256 { return Err(format!("{path} is not the reference libcublasLt (sha256 {sha})")); }
        Ok(())
    }

    /// Load a verified library in the current context.
    pub fn init(path: &str, sms: i32) -> Result<Self> {
        let c = CString::new(path).unwrap();
        let mut err = vec![0u8; 1024];
        let version = unsafe { dsv4_lt_init(c.as_ptr(), err.as_mut_ptr().cast(), err.len()) };
        if version == 0 { return Err(message(&err)); }
        Ok(Self { workspace: Buffer::alloc(WORKSPACE)?, sms, streams: vec![] })
    }

    /// Load the kernels cuBLASLt picks for a GEMM shape (n, k, fp32 or bf16 out) before the first
    /// forward needs them: the library loads them lazily, and the first lm_head call otherwise
    /// takes seconds. Runs the shape once for every m in `ms` on scratch buffers, with its own
    /// workspace and stream (shapes can warm on several threads at once). The heuristic is a pure
    /// function of the problem, so later calls choose exactly as before.
    /// Returns (m, ms, algorithm id) per call.
    pub fn warm(&self, n: usize, k: usize, out_f32: bool, ms: &[usize]) -> Result<Vec<(usize, f64, i32)>> {
        let m_max = ms.iter().copied().max().unwrap_or(1);
        let (w, x, out, ws) = (Buffer::alloc(n * k * 2)?, Buffer::alloc(m_max * k * 2)?, Buffer::alloc(m_max * n * 4)?, Buffer::alloc(WORKSPACE)?);
        let s = Stream::new()?;
        let mut calls = Vec::new();
        for &m in ms {
            let t = std::time::Instant::now();
            let mut info = [0i32; 5];
            let mut err = vec![0u8; 1024];
            let rc = unsafe { dsv4_lt_linear(m as i32, n as i32, k as i32, x.ptr, w.ptr, out.ptr, out_f32 as i32, ws.ptr, WORKSPACE, self.sms,
                                             s.0, info.as_mut_ptr(), err.as_mut_ptr().cast(), err.len()) };
            if rc != 0 { return Err(format!("cuBLASLt warm-up: {}", message(&err))); }
            s.synchronize()?;
            calls.push((m, t.elapsed().as_secs_f64() * 1e3, info[0]));
        }
        Ok(calls)
    }

    /// Give stream `s` its own workspace (same size: the heuristic sees the same problem).
    pub fn add_stream(&mut self, s: Stream) -> Result<()> {
        if !self.streams.iter().any(|(p, _)| *p == s.0 as usize) { self.streams.push((s.0 as usize, Buffer::alloc(WORKSPACE)?)); }
        Ok(())
    }

    /// out [m, n] = x [m, k] . w [n, k]^T (bf16 inputs; `out_f32` selects an fp32 output, as
    /// torch.mm(..., out_dtype=torch.float32)). Returns the chosen algorithm (id, tile, ...).
    pub fn linear(&self, s: Stream, m: usize, n: usize, k: usize, x: u64, w: u64, out: u64, out_f32: bool) -> Result<[i32; 5]> {
        let mut info = [0i32; 5];
        let mut err = vec![0u8; 1024];
        let ws = self.streams.iter().find(|(p, _)| *p == s.0 as usize).map(|(_, b)| b.ptr).unwrap_or(self.workspace.ptr);
        let rc = unsafe { dsv4_lt_linear(m as i32, n as i32, k as i32, x, w, out, out_f32 as i32, ws, WORKSPACE, self.sms,
                                         s.0, info.as_mut_ptr(), err.as_mut_ptr().cast(), err.len()) };
        if rc != 0 { return Err(format!("cuBLASLt: {}", message(&err))); }
        Ok(info)
    }
}
