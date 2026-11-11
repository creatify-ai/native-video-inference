// FlashMLA sparse decode (sgl-project/FlashMLA 5ddc685e, csrc_legacy/api/sparse_decode.cpp
// `sparse_attn_decode_interface`, the SM100 head-64 path) without its torch tensor layer: the same
// parameter struct, split-KV decision, scheduling metadata and combine. FlashMLA's own launchers
// (instantiations/v4_h64*.cu, smxx/decode/{get_decoding_sched_meta,combine}) are compiled into the
// binary; kernels/redirect.cpp re-issues their launches from the reference's CUBINs.
#include <cstdio>
#include <exception>

#include "kernels/params.h"
#include "kernels/sm100/decode/sparse/head64/kernel.h"
#include "kernels/smxx/decode/combine/combine.h"
#include "kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"

extern "C" {

struct Dsv4DecodeArgs {
  int b, s_q, h_q, topk, num_blocks, page_block_size;
  float sm_scale;
  void *q, *kv, *indices, *topk_length, *attn_sink, *lse, *out;
  int extra_num_blocks, extra_page_block_size, extra_topk;
  void *extra_kv, *extra_indices, *extra_topk_length;
  int stride_q_b, stride_q_s_q, stride_q_h_q, stride_kv_block, stride_kv_row, stride_indices_b, stride_indices_s_q;
  int stride_lse_b, stride_lse_s_q, stride_o_b, stride_o_s_q, stride_o_h_q;
  int stride_extra_kv_block, stride_extra_kv_row, stride_extra_indices_b, stride_extra_indices_s_q;
  void* stream;
  int num_sms;
  // Split-KV: scheduling metadata [num_sm_parts] and num_splits [b + 1] (computed here when
  // sched_ready is 0, else reused: SGLang's FlashMLASchedMeta is built by the first layer of a
  // forward that needs it and shared by the rest), then scratch for lse_accum / o_accum.
  void* sched;
  void* num_splits;
  int sched_ready;
  void* scratch;
  size_t scratch_bytes;
};

// Returns 0 on success, else writes err.
int dsv4_flashmla_sparse_decode(const Dsv4DecodeArgs* a, char* err, size_t cap) {
  try {
    using bf16 = cutlass::bfloat16_t;
    constexpr double LOG_2_E = 1.44269504088896340736;
    const int d_qk = 512, d_v = 512;
    // Split-KV only pays off when a request has enough work.
    const bool enable_split_kv = !(a->topk + a->extra_topk <= 640);
    // Decode_Sm100_Head64_Impl::get_meta
    const int num_sm_parts = std::max(a->num_sms / a->s_q, 1);
    const int fixed_overhead_num_blocks = 5, block_size_topk = 64;
    const bool have_extra = a->extra_kv != nullptr;
    SparseAttnDecodeParams params = {
        a->b, a->s_q, a->h_q, 1, d_qk, d_v,
        a->sm_scale, (float)(a->sm_scale * LOG_2_E),
        a->num_blocks, a->page_block_size, a->topk,
        ModelType::V4, ModelType::V4,
        (bf16*)a->q, (bf16*)a->kv, (int*)a->indices, (int*)a->topk_length, (float*)a->attn_sink,
        (float*)a->lse, (bf16*)a->out,
        a->extra_num_blocks, a->extra_page_block_size, a->extra_topk,
        (bf16*)a->extra_kv, (int*)a->extra_indices, (int*)a->extra_topk_length,
        a->stride_q_b, a->stride_q_s_q, a->stride_q_h_q,
        a->stride_kv_block, a->stride_kv_row,
        a->stride_indices_b, a->stride_indices_s_q,
        a->stride_lse_b, a->stride_lse_s_q,
        a->stride_o_b, a->stride_o_s_q, a->stride_o_h_q,
        have_extra ? a->stride_extra_kv_block : 0, have_extra ? a->stride_extra_kv_row : 0,
        have_extra ? a->stride_extra_indices_b : 0, have_extra ? a->stride_extra_indices_s_q : 0,
        (cudaStream_t)a->stream,
        enable_split_kv,
    };
    if (enable_split_kv) {
      char* p = (char*)a->scratch;
      auto take = [&](size_t bytes) { char* r = p; p += (bytes + 1023) / 1024 * 1024; return r; };
      auto* sched = (DecodingSchedMeta*)a->sched;
      auto* num_splits = (int*)a->num_splits;
      const int total_num_splits = a->b + num_sm_parts;
      auto* lse_accum = (float*)take((size_t)total_num_splits * a->s_q * a->h_q * sizeof(float));
      auto* o_accum = (float*)take((size_t)total_num_splits * a->s_q * a->h_q * d_v * sizeof(float));
      if ((size_t)(p - (char*)a->scratch) > a->scratch_bytes) { snprintf(err, cap, "decode scratch too small"); return 1; }
      if (!a->sched_ready) {
        GetDecodeSchedMetaParams sched_params = {
            a->b, a->s_q, block_size_topk, fixed_overhead_num_blocks, a->topk, a->extra_topk,
            (int*)a->topk_length, (int*)a->extra_topk_length, nullptr, sched, num_splits, num_sm_parts,
            (cudaStream_t)a->stream};
        legacy::smxx::decode::run_get_decoding_sched_meta_kernel(sched_params);
      }
      params.tile_scheduler_metadata_ptr = sched;
      params.num_splits_ptr = num_splits;
      params.num_sm_parts = num_sm_parts;
      params.lse_accum = lse_accum;
      params.o_accum = o_accum;
      params.stride_lse_accum_split = a->s_q * a->h_q;
      params.stride_lse_accum_s_q = a->h_q;
      params.stride_o_accum_split = a->s_q * a->h_q * d_v;
      params.stride_o_accum_s_q = a->h_q * d_v;
      params.stride_o_accum_h_q = d_v;
      legacy::sm100::decode::sparse::head64::run_flash_splitkv_mla_fp8_sparse_kernel<
          legacy::sm100::decode::sparse::head64::Config{ModelType::V4, ModelType::V4, true}>(params);
      CombineParams combine_params = {
          a->b, a->s_q, a->h_q, d_v,
          params.lse, params.out,
          params.stride_lse_b, params.stride_lse_s_q,
          params.stride_o_b, params.stride_o_s_q, params.stride_o_h_q,
          params.lse_accum, params.o_accum,
          params.stride_lse_accum_split, params.stride_lse_accum_s_q,
          params.stride_o_accum_split, params.stride_o_accum_s_q, params.stride_o_accum_h_q,
          params.tile_scheduler_metadata_ptr, params.num_splits_ptr, params.num_sm_parts,
          (float*)a->attn_sink, (cudaStream_t)a->stream};
      legacy::smxx::decode::run_flash_mla_combine_kernel<bf16>(combine_params);
    } else {
      legacy::sm100::decode::sparse::head64::run_flash_splitkv_mla_fp8_sparse_kernel<
          legacy::sm100::decode::sparse::head64::Config{ModelType::V4, ModelType::V4, false}>(params);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { snprintf(err, cap, "FlashMLA decode launch: %s", cudaGetErrorString(e)); return 1; }
    return 0;
  } catch (std::exception const& ex) {
    snprintf(err, cap, "%s", ex.what());
    return 1;
  }
}

}  // extern "C"
