//! Load-time layout of the FP8 block-quantized dense weights, as SGLang prepares them for DeepGEMM
//! on SM100: Q/KV-A rows fused (`wqkv_a`), shared-expert gate/up fused (`gate_up_proj`), and each
//! 128x128 UE8M0 block scale broadcast to its rows and packed four K-blocks per int32, MN-major
//! (`get_mn_major_tma_aligned_packed_ue8m0_tensor` after `transform_sf_into_required_layout`).
//!
//! The checkpoint stores the scales as E8M0 bytes. SGLang casts them to FP32 (2^(e-127)) and the
//! packer takes `bits >> 23`, which returns the same byte for every E8M0 value, so the bytes pass
//! through unchanged.

/// TMA alignment of the MN dimension for 4-byte elements (16 bytes).
pub fn tma_aligned_mn(mn: usize) -> usize { mn.div_ceil(4) * 4 }

/// Packed scales of one group: `scales` is the checkpoint's [ceil(mn/128), sf_k] E8M0 bytes.
/// Returns (logical [mn, ceil(sf_k/4)] int32 values row-major, memory image MN-major padded).
pub fn pack_block_scales(scales: &[u8], mn: usize, sf_k: usize, gran_mn: usize) -> (Vec<u32>, Vec<u32>) {
    let packed_k = sf_k.div_ceil(4);
    let aligned = tma_aligned_mn(mn);
    let mut logical = vec![0u32; mn * packed_k];
    let mut memory = vec![0u32; aligned * packed_k];
    for m in 0..mn {
        let row = m / gran_mn;
        for j in 0..packed_k {
            let mut v = 0u32;
            for b in 0..4 {
                let k = 4 * j + b;
                if k < sf_k { v |= (scales[row * sf_k + k] as u32) << (8 * b); }
            }
            logical[m * packed_k + j] = v;
            memory[j * aligned + m] = v;
        }
    }
    (logical, memory)
}

/// Row-concatenate two row-major matrices with the same row width.
pub fn cat_rows(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

pub fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

/// float8_e4m3fn -> f32 (exact).
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f32;
    if e == 15 && (b & 7) == 7 { return f32::NAN; }
    if e == 0 { sign * m * (2.0f32).powi(-9) } else { sign * (1.0 + m / 8.0) * (2.0f32).powi(e - 7) }
}

/// c10 `fp8e4m3fn_from_fp32_value`: round to nearest even, >= 480 becomes NaN.
pub fn f32_to_e4m3(f: f32) -> u8 {
    let mut bits = f.to_bits();
    let sign = bits & 0x8000_0000;
    bits ^= sign;
    const FP8_MAX: u32 = 1087 << 20;
    const DENORM_MASK: u32 = 141 << 23;
    let result: u8 = if bits >= FP8_MAX {
        0x7f
    } else if bits < (121 << 23) {
        let v = f32::from_bits(bits) + f32::from_bits(DENORM_MASK);
        (v.to_bits().wrapping_sub(DENORM_MASK)) as u8
    } else {
        let mant_odd = (bits >> 20) & 1;
        let mut b = bits.wrapping_add(((7u32.wrapping_sub(127)) << 23).wrapping_add(0x7FFFF));
        b = b.wrapping_add(mant_odd);
        (b >> 20) as u8
    };
    result | (sign >> 24) as u8
}

/// c10 `round_to_nearest_even` f32 -> bf16 bits.
pub fn f32_to_bf16(f: f32) -> u16 {
    if f.is_nan() { return 0x7fc0; }
    let u = f.to_bits();
    let bias = ((u >> 16) & 1) + 0x7fff;
    (u.wrapping_add(bias) >> 16) as u16
}

pub fn bf16_to_f32(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }

/// float8_e8m0fnu -> f32: 2^(e-127); 0 is the subnormal 2^-127, 255 is NaN.
pub fn e8m0_to_f32(e: u8) -> f32 {
    match e { 0 => f32::from_bits(0x0040_0000), 255 => f32::NAN, _ => f32::from_bits((e as u32) << 23) }
}

/// `ceil_to_ue8m0`: the next power of two at or above |x|, exponent clamped to [1, 254].
pub fn ceil_to_ue8m0(x: f32) -> f32 {
    let bits = x.abs().to_bits();
    let mut exp = ((bits >> 23) & 0xff) as i32;
    if bits & 0x7f_ffff != 0 { exp += 1; }
    f32::from_bits((exp.clamp(1, 254) as u32) << 23)
}

/// SGLang's `requant_weight_ue8m0` for one [n, k] FP8 matrix with 128x128 E8M0 block scales:
/// dequantize to BF16, re-quantize per 128x128 block with power-of-two scales. Returns the new
/// FP8 bytes and the new scales as E8M0 bytes ([ceil(n/128), ceil(k/128)]).
pub fn requant_ue8m0(weight: &[u8], scales: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let (bn, bk) = (n.div_ceil(128), k.div_ceil(128));
    let sk = bk; // checkpoint scale columns
    // 1. block_quant_dequant(..., bfloat16): (fp8 as f32) * scale_f32, rounded to bf16.
    let mut deq = vec![0u16; n * k];
    for r in 0..n {
        for c in 0..k {
            let s = e8m0_to_f32(scales[(r / 128) * sk + c / 128]);
            deq[r * k + c] = f32_to_bf16(e4m3_to_f32(weight[r * k + c]) * s);
        }
    }
    // 2. per_block_cast_to_fp8 (zero padding never raises a block's amax).
    let mut out = vec![0u8; n * k];
    let mut new_scales = vec![0u8; bn * bk];
    for i in 0..bn {
        for j in 0..bk {
            let mut amax = 0f32;
            for r in i * 128..((i + 1) * 128).min(n) {
                for c in j * 128..((j + 1) * 128).min(k) {
                    amax = amax.max(bf16_to_f32(deq[r * k + c]).abs());
                }
            }
            let sf = ceil_to_ue8m0(amax.max(1e-4f32) / 448.0);
            let inv = 1.0f32 / sf;
            new_scales[i * bk + j] = (sf.to_bits() >> 23) as u8;
            for r in i * 128..((i + 1) * 128).min(n) {
                for c in j * 128..((j + 1) * 128).min(k) {
                    out[r * k + c] = f32_to_e4m3(bf16_to_f32(deq[r * k + c]) * inv);
                }
            }
        }
    }
    (out, new_scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_round_trips_every_finite_code() {
        for b in 0u8..=255 {
            if b & 0x7f == 0x7f { continue; } // NaN
            assert_eq!(f32_to_e4m3(e4m3_to_f32(b)), b, "code {b:#x}");
        }
        assert_eq!(f32_to_e4m3(448.0), 0x7e);
        assert_eq!(f32_to_e4m3(1000.0), 0x7f);
    }

    #[test]
    fn requant_keeps_values_whose_scale_is_already_minimal() {
        // One 128x128 block whose amax is exactly 448 * 2^-3: the scale stays 2^-3 (E8M0 124).
        let mut w = vec![0u8; 128 * 128];
        w[5] = 0x7e; // 448
        w[6] = 0x38; // 1.0
        let (out, s) = requant_ue8m0(&w, &[124], 128, 128);
        assert_eq!(s, vec![124]);
        assert_eq!(out, w);
    }

    #[test]
    fn packs_four_k_blocks_little_endian_and_broadcasts_rows() {
        // mn = 256 rows (2 row blocks), sf_k = 5 -> 2 packed columns, the second zero padded.
        let scales: Vec<u8> = (0..10).map(|i| 100 + i as u8).collect();
        let (logical, memory) = pack_block_scales(&scales, 256, 5, 128);
        assert_eq!(logical[0], u32::from_le_bytes([100, 101, 102, 103]));
        assert_eq!(logical[1], 104);
        assert_eq!(logical[200 * 2], u32::from_le_bytes([105, 106, 107, 108]));
        assert_eq!(memory[256 + 200], 109); // column 1 of row 200, MN-major
    }
}
