// FlashMLA legacy host code (sgl-project/FlashMLA 5ddc685e, csrc_legacy) that builds the kernels'
// by-value parameter structs: the same CuTe `make_tma_copy` calls as the reference's `run()`, so
// the TmaParams bytes match the reference's for every meaningful field. Nothing here launches a
// kernel: the native port launches the reference's own CUBIN with these bytes.
#include <cstring>
#include <cmath>

#include "kernels/params.h"
#include "kernels/sm100/prefill/sparse/fwd/head64/phase1.cuh"
#include "kernels/sm100/decode/sparse/head64/kernel.cuh"

namespace {

template <typename T>
size_t copy_out(const T& value, void* out, size_t cap) {
    if (sizeof(T) > cap) return 0;
    std::memcpy(out, &value, sizeof(T));
    return sizeof(T);
}

// legacy::sm100::prefill::sparse_fwd::head64::KernelTemplate<Prefill, 512>::run, without the launch.
template <SparseAttnFwdMode FWD_MODE, int D_QK>
size_t prefill_tma(const SparseAttnFwdParams& params, void* out, size_t cap) {
    using K = legacy::sm100::prefill::sparse_fwd::head64::KernelTemplate<FWD_MODE, D_QK>;
    using namespace cute;
    using bf16 = cutlass::bfloat16_t;
    constexpr int D_V = K::D_V, D_Q = K::D_Q;
    auto shape_Q_nope = make_shape(params.h_q, D_V, params.s_q);
    auto tma_Q_nope = make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((bf16*)params.q), make_layout(shape_Q_nope, make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q))),
        typename K::SmemLayoutQNoPE{});
    auto shape_Q_rope = make_shape(params.h_q, D_Q - D_V == 0 ? 64 : D_Q - D_V, params.s_q);
    auto tma_Q_rope = make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((bf16*)params.q + D_V), make_layout(shape_Q_rope, make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q))),
        typename K::SmemLayoutQRoPE{});
    auto shape_O = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_O = make_tma_copy(SM90_TMA_STORE{},
        make_tensor(make_gmem_ptr((bf16*)params.out), make_layout(shape_O, make_stride(params.d_v, _1{}, params.h_q * params.d_v))),
        typename K::template SmemLayoutOTiles<1>{});
    CUtensorMap tensor_map_kv_nope;
    {
        uint64_t size[2] = {(uint64_t)D_V, (unsigned long)params.s_kv};
        uint64_t stride[1] = {params.stride_kv_s_kv * sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        if (cuTensorMapEncodeTiled(&tensor_map_kv_nope, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, params.kv, size, stride, box_size,
                                   elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
            return 0;
    }
    typename K::template TmaParams<decltype(shape_Q_nope), decltype(tma_Q_nope), decltype(shape_Q_rope), decltype(tma_Q_rope),
                                   decltype(shape_O), decltype(tma_O)> tma_params = {
        shape_Q_nope, tma_Q_nope, shape_Q_rope, tma_Q_rope, shape_O, tma_O, tensor_map_kv_nope};
    return copy_out(tma_params, out, cap);
}

}  // namespace

extern "C" {

// SparseAttnFwdParams as sparse_attn_prefill_interface fills it (stream left null).
struct Dsv4PrefillArgs {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    float sm_scale;
    void *q, *kv, *indices, *attn_sink, *topk_length;
    int stride_q_s_q, stride_q_h_q, stride_kv_s_kv, stride_kv_h_kv, stride_indices_s_q, stride_indices_h_kv;
    void *out, *max_logits, *lse;
    int num_sm;
    int stats_in_log2;
};

static SparseAttnFwdParams prefill_params(const Dsv4PrefillArgs* a) {
    constexpr double LOG_2_E = 1.44269504088896340736;
    SparseAttnFwdParams p = {
        a->s_q, a->s_kv, a->h_q, a->h_kv, a->d_qk, a->d_v, a->topk,
        a->sm_scale, (float)((double)a->sm_scale * LOG_2_E),
        (cutlass::bfloat16_t*)a->q, (cutlass::bfloat16_t*)a->kv, (int*)a->indices, (float*)a->attn_sink, (int*)a->topk_length,
        a->stride_q_s_q, a->stride_q_h_q, a->stride_kv_s_kv, a->stride_kv_h_kv, a->stride_indices_s_q, a->stride_indices_h_kv,
        (cutlass::bfloat16_t*)a->out, (float*)a->max_logits, (float*)a->lse,
        a->num_sm, a->stats_in_log2 != 0, nullptr};
    return p;
}

// Writes the kernel's two by-value parameters (SparseAttnFwdParams, TmaParams); returns 0 on error.
int dsv4_flashmla_prefill_head64_d512(const Dsv4PrefillArgs* a, void* params_out, size_t params_cap, size_t* params_size,
                                      void* tma_out, size_t tma_cap, size_t* tma_size, size_t* smem_size) {
    SparseAttnFwdParams p = prefill_params(a);
    *params_size = copy_out(p, params_out, params_cap);
    *tma_size = prefill_tma<SparseAttnFwdMode::Prefill, 512>(p, tma_out, tma_cap);
    using K = legacy::sm100::prefill::sparse_fwd::head64::KernelTemplate<SparseAttnFwdMode::Prefill, 512>;
    *smem_size = sizeof(typename K::SharedMemoryPlan);
    return *params_size && *tma_size;
}

}  // extern "C"
