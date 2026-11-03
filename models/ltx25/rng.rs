//! The CUDA default generator as torch.manual_seed(seed) leaves it, and torch.randn on it.
use crate::cuda::{check, Stream};
use crate::Result;
use std::ffi::{c_int, c_uint, c_void};

extern "C" {
    fn ltx25_randn(out: *mut c_void, bf16: c_int, numel: i64, seed: u64, offset: u64, grid: c_uint, stream: Stream) -> c_int;
    fn cudaDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: c_int) -> c_int;
}

/// Philox seed and per-thread offset of CUDAGeneratorImpl.
#[derive(Clone, Copy, Debug)]
pub struct CudaGenerator { pub seed: u64, pub offset: u64, sms: u32, threads_per_sm: u32 }

impl CudaGenerator {
    /// torch.cuda.manual_seed(seed): offset 0. Launch geometry follows device 0's SM count.
    pub fn new(seed: u64) -> Result<Self> {
        let (mut sms, mut threads) = (0, 0);
        check(unsafe { cudaDeviceGetAttribute(&mut sms, 16, 0) }, "multiProcessorCount")?;
        check(unsafe { cudaDeviceGetAttribute(&mut threads, 39, 0) }, "maxThreadsPerMultiProcessor")?;
        Ok(Self { seed, offset: 0, sms: sms as u32, threads_per_sm: threads as u32 })
    }

    /// calc_execution_policy: (counter_offset, grid) for `numel` outputs, unroll 4.
    fn policy(&self, numel: u64) -> (u64, u32) {
        let block = 256u64;
        let blocks_per_sm = (self.threads_per_sm / 256) as u64;
        let grid = numel.div_ceil(block).min(self.sms as u64 * blocks_per_sm);
        let counter = ((numel - 1) / (block * grid * 4) + 1) * 4;
        (counter, grid as u32)
    }

    /// torch.randn into a contiguous device tensor of `numel` bf16 (or fp32) values.
    pub fn randn(&mut self, out: u64, numel: usize, bf16: bool, stream: Stream) -> Result<()> {
        if numel == 0 { return Ok(()); }
        let (counter, grid) = self.policy(numel as u64);
        let offset = self.offset;
        // CUDAGeneratorState::increase rounds the increment up to a multiple of 4.
        self.offset += counter.div_ceil(4) * 4;
        check(unsafe { ltx25_randn(out as *mut c_void, c_int::from(bf16), i64::try_from(numel)?, self.seed, offset, grid, stream) },
              "randn")
    }
}
