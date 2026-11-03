//! The pinned MT19937 and bulk FP32 normal arithmetic.
use crate::Result;
#[cfg(feature="avx-noise")]
extern "C" { fn infra_normal_transform16(data: *mut f32, count: usize); }

/// MT19937 follows the pinned PyTorch engine; see NOTICE.md for attribution.
pub struct CpuGenerator { state: [u32; 624], next: usize, pub draws: usize }
impl CpuGenerator {
    pub fn new(seed: u64) -> Self {
        let mut state = [0; 624];
        state[0] = seed as u32;
        for i in 1..624 {
            state[i] = 1812433253u32.wrapping_mul(state[i-1] ^ (state[i-1] >> 30)).wrapping_add(i as u32);
        }
        Self { state, next: 624, draws: 0 }
    }
    pub fn random(&mut self) -> u32 {
        if self.next == 624 {
            for i in 0..624 {
                let joined = (self.state[i] & 0x80000000) | (self.state[(i+1)%624] & 0x7fffffff);
                self.state[i] = self.state[(i+397)%624] ^ (joined >> 1) ^ if joined & 1 != 0 { 0x9908b0df } else { 0 };
            }
            self.next = 0;
        }
        let mut y = self.state[self.next];
        self.next += 1; self.draws += 1;
        y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >> 18;
        y
    }
    pub fn uniform(&mut self) -> f32 { (self.random() & 0x00ffffff) as f32 * (1.0 / 16777216.0) }
    pub fn normal(&mut self, count: usize) -> Result<Vec<f32>> {
        if count < 16 { return Err("only the original contiguous FP32 bulk normal path is implemented".into()); }
        #[cfg(any(target_arch="x86",target_arch="x86_64"))]
        let supported=std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
        #[cfg(not(any(target_arch="x86",target_arch="x86_64")))]
        let supported=false;
        if !supported || !cfg!(feature="avx-noise") {
            return Err("pinned CPU normal path requires AVX2 and FMA".into());
        }
        let mut values: Vec<f32> = (0..count).map(|_| self.uniform()).collect();
        #[cfg(feature="avx-noise")]
        unsafe { infra_normal_transform16(values.as_mut_ptr(), count / 16 * 16); }
        if count % 16 != 0 {
            for value in &mut values[count-16..] { *value = self.uniform(); }
            #[cfg(feature="avx-noise")]
            unsafe { infra_normal_transform16(values.as_mut_ptr().add(count-16), 16); }
        }
        Ok(values)
    }
}

