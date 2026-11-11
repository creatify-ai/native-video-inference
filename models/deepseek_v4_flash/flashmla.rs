//! FlashMLA sparse attention (sgl-flash-mla legacy SM100 kernels): parameter structs from the
//! host shim (kernels/flashmla_host.cu), launches of the reference CUBINs.
use crate::cuda::Stream;
use crate::runtime::{Kernels, Launch};
use crate::Result;
use std::ffi::c_void;

#[repr(C)]
pub struct PrefillArgs {
    pub s_q: i32, pub s_kv: i32, pub h_q: i32, pub h_kv: i32, pub d_qk: i32, pub d_v: i32, pub topk: i32,
    pub sm_scale: f32,
    pub q: u64, pub kv: u64, pub indices: u64, pub attn_sink: u64, pub topk_length: u64,
    pub stride_q_s_q: i32, pub stride_q_h_q: i32, pub stride_kv_s_kv: i32, pub stride_kv_h_kv: i32,
    pub stride_indices_s_q: i32, pub stride_indices_h_kv: i32,
    pub out: u64, pub max_logits: u64, pub lse: u64,
    pub num_sm: i32,
    pub stats_in_log2: i32,
}

extern "C" {
    fn dsv4_flashmla_prefill_head64_d512(a: *const PrefillArgs, params: *mut c_void, params_cap: usize, params_size: *mut usize,
                                         tma: *mut c_void, tma_cap: usize, tma_size: *mut usize, smem: *mut usize) -> i32;
}

const PREFILL_HEAD64: &str = "_ZN6legacy5sm1007prefill10sparse_fwd6head6422sparse_attn_fwd_kernelINS3_14KernelTemplateIL17SparseAttnFwdMode0ELi512EEENS7_9TmaParamsIN4cute5tupleIJiiiEEENS9_9TiledCopyINS9_9Copy_AtomIJNS9_11Copy_TraitsINS9_13SM90_TMA_LOADEJNS9_1CILi65536EEENS9_12AuxTmaParamsINSA_IJNS9_11ScaledBasisINSG_ILi1EEEJLi1EEEENSJ_ISK_JLi0EEEENSJ_ISK_JLi2EEEEEEERKNS9_6LayoutINSA_IJNSG_ILi64EEESQ_SK_EEESO_EERKNS9_7SwizzleILi3ELi4ELi3EEEEEEEEN7cutlass10bfloat16_tEEEENSP_INSA_IJSK_NSA_IJNSA_IJNSA_IJSQ_SQ_EEENSG_ILi8EEEEEEEEEEEENSA_IJNSG_ILi0EEENSA_IJNSA_IJNSA_IJSQ_SK_EEENSG_ILi4096EEEEEEEEEEEEEENSA_IJSQ_NSG_ILi512EEEEEEEESB_NSC_INSD_IJNSE_ISF_JNSG_ILi32768EEENSI_ISO_RKNSP_INSA_IJNSG_ILi32EEESQ_SK_EEESO_EERKNSV_ILi2ELi4ELi3EEEEEEEES12_EEENSP_INSA_IJSK_NSA_IJNSA_IJNSA_IJS1K_SQ_EEES19_EEEEEEEEENSA_IJS19_NSA_IJNSA_IJS1A_NSG_ILi2048EEEEEEEEEEEEEENSA_IJSQ_S19_EEEEESB_NSC_INSD_IJNSE_INS9_14SM90_TMA_STOREEJSH_SZ_EEES12_EEENSP_INSA_IJSK_NSA_IJNSA_IJS14_SK_EEEEEEEEENSA_IJS19_NSA_IJNSA_IJS1A_S19_EEEEEEEEEEES14_EEEEEEv19SparseAttnFwdParamsT0_";

/// `flash_mla_sparse_fwd` (legacy SM100, 64 heads, d_qk 512) for one prefill chunk.
pub fn sparse_prefill(k: &Kernels, s: Stream, a: &PrefillArgs) -> Result<()> {
    let mut params = vec![0u8; 512];
    let mut tma = vec![0u8; 4096];
    let (mut ps, mut ts, mut smem) = (0usize, 0usize, 0usize);
    let ok = unsafe {
        dsv4_flashmla_prefill_head64_d512(a, params.as_mut_ptr().cast(), params.len(), &mut ps, tma.as_mut_ptr().cast(),
                                          tma.len(), &mut ts, &mut smem)
    };
    if ok == 0 { return Err("FlashMLA prefill host shim failed".into()); }
    params.truncate(ps);
    tma.truncate(ts);
    Launch { name: PREFILL_HEAD64, grid: [a.s_q as u32, 1, 1], block: [384, 1, 1], smem: smem as u32, attrs: vec![],
             params: vec![params, tma] }.run(k, s)
}

#[repr(C)]
pub struct DecodeArgs {
    pub b: i32, pub s_q: i32, pub h_q: i32, pub topk: i32, pub num_blocks: i32, pub page_block_size: i32,
    pub sm_scale: f32,
    pub q: u64, pub kv: u64, pub indices: u64, pub topk_length: u64, pub attn_sink: u64, pub lse: u64, pub out: u64,
    pub extra_num_blocks: i32, pub extra_page_block_size: i32, pub extra_topk: i32,
    pub extra_kv: u64, pub extra_indices: u64, pub extra_topk_length: u64,
    pub stride_q_b: i32, pub stride_q_s_q: i32, pub stride_q_h_q: i32, pub stride_kv_block: i32, pub stride_kv_row: i32,
    pub stride_indices_b: i32, pub stride_indices_s_q: i32,
    pub stride_lse_b: i32, pub stride_lse_s_q: i32, pub stride_o_b: i32, pub stride_o_s_q: i32, pub stride_o_h_q: i32,
    pub stride_extra_kv_block: i32, pub stride_extra_kv_row: i32, pub stride_extra_indices_b: i32, pub stride_extra_indices_s_q: i32,
    pub stream: *mut c_void,
    pub num_sms: i32,
    pub sched: u64,
    pub num_splits: u64,
    pub sched_ready: i32,
    pub scratch: u64,
    pub scratch_bytes: usize,
}

extern "C" {
    fn dsv4_flashmla_sparse_decode(a: *const DecodeArgs, err: *mut std::ffi::c_char, cap: usize) -> i32;
}

/// FlashMLA sparse decode (FP8 paged caches, optional extra compressed cache; split-KV with
/// scheduling metadata and combine when topk + extra_topk > 640), via kernels/flashmla_decode.cu.
pub fn sparse_decode(a: &DecodeArgs) -> Result<()> {
    let mut err = vec![0u8; 2048];
    let rc = unsafe { dsv4_flashmla_sparse_decode(a, err.as_mut_ptr().cast(), err.len()) };
    if rc != 0 {
        let n = err.iter().position(|&b| b == 0).unwrap_or(err.len());
        return Err(format!("FlashMLA decode: {}", String::from_utf8_lossy(&err[..n])));
    }
    Ok(())
}
