//! Load-time layout of the NVFP4 routed experts for the trtllm-gen MoE cubins, as SGLang's
//! `align_fp4_moe_weights_for_flashinfer_trtllm` / flashinfer's `prepare_static_weights_for_trtllm_fp4_moe`
//! produce it: gated row interleave, 32-row block shuffle (epilogue tile 128) and the 128x4
//! swizzle of the FP8 block scales. Pure index maps; no arithmetic touches the values.

/// `get_reorder_rows_for_gated_act_gemm_row_indices`: [r0..r(M/2-1), rM/2..] -> r0, rM/2, r1, ...
pub fn gated_row_order(m: usize) -> Vec<usize> {
    let half = m.div_ceil(2);
    let mut out = vec![0; m];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = if i % 2 == 0 { i / 2 } else { half + i / 2 };
    }
    out
}

const BLK32: [usize; 32] = [0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27,
                            4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31];
const BLK16: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

/// `get_shuffle_matrix_a_row_indices`: row_indices[new_row] = old_row.
pub fn shuffle_rows(m: usize, epilogue_tile_m: usize) -> Vec<usize> {
    let block = if epilogue_tile_m % 128 == 0 { 32 } else { 16 };
    assert_eq!(m % block, 0, "rows must be a multiple of the shuffle block");
    let map: &[usize] = if block == 32 { &BLK32 } else { &BLK16 };
    let mut out = vec![0; m];
    for old in 0..m {
        let new = old / block * block + map[old % block];
        out[new] = old;
    }
    out
}

/// GEMM1 (w3/w1 fused) rows: gated reorder, then the shuffle (`permute0[permute1]`).
pub fn gemm1_rows(m: usize, epilogue_tile_m: usize) -> Vec<usize> {
    let p0 = gated_row_order(m);
    shuffle_rows(m, epilogue_tile_m).into_iter().map(|i| p0[i]).collect()
}

/// Gather rows of a row-major [rows, row_bytes] matrix.
pub fn gather_rows(src: &[u8], row_bytes: usize, rows: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows.len() * row_bytes);
    for &r in rows {
        out.extend_from_slice(&src[r * row_bytes..(r + 1) * row_bytes]);
    }
    out
}

/// Size of the 128x4-swizzled scale buffer for an [m, k] scale matrix.
pub fn swizzled_len(m: usize, k: usize) -> usize { m.div_ceil(128) * 128 * k.div_ceil(4) * 4 }

/// `block_scale_interleave` (one batch): [m, k] row-major scales -> SWIZZLED_128x4, zero padded.
pub fn interleave_scales(src: &[u8], m: usize, k: usize) -> Vec<u8> {
    let tiles_k = k.div_ceil(4);
    let mut out = vec![0u8; swizzled_len(m, k)];
    for mi in 0..m.div_ceil(128) * 128 {
        for ki in 0..tiles_k * 4 {
            let v = if mi < m && ki < k { src[mi * k + ki] } else { 0 };
            let dst = ((mi / 128) * tiles_k + ki / 4) * 512 + (mi % 32) * 16 + ((mi % 128) / 32) * 4 + ki % 4;
            out[dst] = v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gated_order_interleaves_halves() {
        assert_eq!(gated_row_order(8), vec![0, 4, 1, 5, 2, 6, 3, 7]);
    }

    #[test]
    fn shuffle_is_a_permutation_with_the_reference_block_map() {
        let p = shuffle_rows(64, 128);
        let mut sorted = p.clone();
        sorted.sort();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
        // new row BLK32[old] holds old row: old 1 -> new 8
        assert_eq!(p[8], 1);
        assert_eq!(p[32 + 8], 33);
    }

    #[test]
    fn interleave_places_the_first_tile_like_the_kernel() {
        let (m, k) = (128, 4);
        let src: Vec<u8> = (0..m * k).map(|i| (i % 251) as u8).collect();
        let out = interleave_scales(&src, m, k);
        // row 33, col 2 -> (33 % 32) * 16 + (33 / 32) * 4 + 2 = 22
        assert_eq!(out[22], src[33 * k + 2]);
    }
}
