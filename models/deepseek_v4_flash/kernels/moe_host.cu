// flashinfer's trtllm-gen FP4 MoE (`trtllm_fp4_block_scale_moe` with precomputed, unpacked routing:
// the path SGLang's `flashinfer_trtllm_routed` runner takes for DeepSeek-V4's NVFP4 experts), minus
// its TVM-FFI launcher. Everything below `dsv4_moe_fp4_routed` mirrors FusedMoeLauncher /
// FP4BlockScaleLauncher (csrc/trtllm_fused_moe_kernel_launcher.cu) step for step; the routing and
// MoE runners themselves are flashinfer's own sources, compiled with the reference's JIT flags.
//
// Kernels: routing and finalize launch through the CUDA runtime and are re-issued from the kernel
// bundle by kernels/redirect.cpp; the trtllm-gen batched GEMMs are loaded by BatchedGemmInterface
// from `getCubin` below, which serves flashinfer-cubin's own files (named by their SHA-256, the
// hash flashinferMetaInfo.h records), exactly the bytes the reference passes to cuModuleLoadData.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/BatchedGemmInterface.h"
#include "flashinfer/trtllm/fused_moe/runner.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tvm/ffi/c_api.h"

namespace btg = batchedGemm::trtllm::gen;
using namespace tensorrt_llm::kernels::trtllmgen_moe;

// The routing sources' TVM_FFI_ICHECK error paths take a backtrace from libtvm_ffi, which this binary
// does not link; errors still propagate as exceptions, with this empty backtrace.
extern "C" const TVMFFIByteArray* TVMFFIBacktrace(const char*, int, const char*, int) {
  static const TVMFFIByteArray empty{"", 0};
  return &empty;
}

namespace flashinfer::trtllm_cubin_loader {
static std::string g_cubin_dir;
std::string getCubin(const std::string& name, const std::string& sha256) {
  std::string const path = g_cubin_dir + "/" + sha256 + ".cubin";
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("trtllm-gen cubin " + name + " (sha256 " + sha256 + ") not found under " + g_cubin_dir);
  std::stringstream data;
  data << f.rdbuf();
  return data.str();
}
}  // namespace flashinfer::trtllm_cubin_loader

namespace {

// Bump allocator over the caller's device scratch; 512-byte blocks like PyTorch's allocator.
struct Scratch {
  char* base;
  size_t cap, used = 0;
  void* take(size_t bytes) {
    size_t start = (used + 511) / 512 * 512;
    used = start + (bytes ? bytes : 1);
    return used <= cap ? base + start : nullptr;
  }
};

std::string kernel_name(int64_t config) {
  auto const* configs = batchedGemm::batchedGemm::BatchedGemmInterface().getBatchedGemmConfigs();
  return configs[config].mFunctionName;
}

}  // namespace

extern "C" {

struct Dsv4MoeArgs {
  int32_t num_tokens, num_experts, top_k, hidden_size, intermediate_size, local_expert_offset, local_num_experts;
  int32_t tile_tokens_dim;
  int32_t enable_pdl;
  const char* gemm1_kernel;  // trtllm-gen kernel names selecting the MoE tactic
  const char* gemm2_kernel;
  void* topk_ids;             // int32 [T, top_k]
  void* topk_weights;         // float32 [T, top_k]
  void* hidden_states;        // packed e2m1 [T, H / 2]
  void* hidden_states_scale;  // e4m3 [T, H / 16], linear
  void* gemm1_weights;
  void* gemm1_weights_scale;
  void* gemm2_weights;
  void* gemm2_weights_scale;
  float* gemm1_clamp_limit;
  float* output1_scales_scalar;
  float* output1_scales_gate_scalar;
  float* output2_scales_scalar;
  void* output;  // bf16 [T, H]
  void* scratch;
  size_t scratch_bytes;
  void* stream;
  const char* cubin_dir;
  size_t* scratch_used;  // out
  // 1: MXFP4 weights with MXFP8 activations (MxE2m1 x MxE4m3, 32-element block scales: the MTP
  // draft's experts); 0: NVFP4 x NVFP4 (16-element block scales).
  int32_t mx;
};

// Returns 0 on success; otherwise writes a message to err.
int dsv4_moe_fp4_routed(const Dsv4MoeArgs* a, char* err, size_t err_cap) {
  try {
    flashinfer::trtllm_cubin_loader::g_cubin_dir = a->cubin_dir;
    cudaStream_t stream = static_cast<cudaStream_t>(a->stream);
    int device = 0;
    cudaGetDevice(&device);
    Scratch scratch{static_cast<char*>(a->scratch), a->scratch_bytes};
    auto take = [&](size_t bytes) {
      void* p = scratch.take(bytes);
      if (!p) throw std::runtime_error("MoE scratch too small (" + std::to_string(a->scratch_bytes) + " bytes)");
      return p;
    };

    // trtllm_fp4_block_scale_moe: arguments of the launcher for this tile.
    MoE::MoERunnerArgs args;
    args.num_tokens = a->num_tokens;
    args.num_experts = a->num_experts;
    args.hidden_size = a->hidden_size;
    args.hidden_size_output = a->hidden_size;  // moe_hidden_size_output(output.size(1)) without valid dims
    args.top_k = a->top_k;
    args.num_fused_shared_experts = 0;
    args.n_group = 0;
    args.topk_group = 0;
    args.local_expert_offset = a->local_expert_offset;
    args.local_num_experts = a->local_num_experts;
    args.intermediate_size = a->intermediate_size;
    args.routed_scaling_factor = 1.0f;
    args.do_finalize = true;
    args.output = a->output;
    args.output_scale = nullptr;
    // FP4BlockScaleLauncher::init + FusedMoeLauncher::init_common
    btg::Dtype const dtype_act = a->mx ? btg::Dtype::MxE4m3 : btg::Dtype::E2m1;
    btg::Dtype const dtype_weights = a->mx ? btg::Dtype::MxE2m1 : btg::Dtype::E2m1;
    args.mDtypeElt = dtype_act;
    args.mDtypeOut = btg::Dtype::Bfloat16;
    args.mUseDeepSeekFp8 = false;
    args.routing_logits = nullptr;
    args.routing_bias = nullptr;
    args.hidden_states = a->hidden_states;
    args.gemm1_weights = a->gemm1_weights;
    args.gemm2_weights = a->gemm2_weights;
    args.gemm1_bias = nullptr;
    args.gemm1_bias_type = batchedGemm::gemm::BiasType::None;
    int32_t const tile = a->tile_tokens_dim;

    // prepare_routing_common + FP4BlockScaleLauncher::prepare_routing
    MoE::MoEWorkspace workspace;
    int32_t const k = args.top_k, e = args.num_experts, t = args.num_tokens;
    auto* num_tokens_per_expert = static_cast<int*>(take(sizeof(int) * e));
    int32_t const max_num_padded_tokens = Routing::getMaxPermutedPaddedCount(t, k, e, tile);
    auto* total_num_padded_tokens = static_cast<int*>(take(sizeof(int)));
    auto* expanded_idx_to_permuted_idx = static_cast<int*>(take(sizeof(int) * t * k));
    auto* permuted_idx_to_token_idx = static_cast<int*>(take(sizeof(int) * (max_num_padded_tokens + 1)));
    auto* expert_indexes = static_cast<int*>(take(sizeof(int) * t * k));
    int64_t const histogram = std::max(e * 2, 256 * 2);
    auto* expert_count_histogram = static_cast<int*>(take(sizeof(int) * histogram));
    int32_t const max_num_ctas = Routing::getMaxNumCtasInBatchDim(t, k, e, tile);
    auto* cta_idx_xy_to_batch_idx = static_cast<int*>(take(sizeof(int) * max_num_ctas));
    auto* cta_idx_xy_to_mn_limit = static_cast<int*>(take(sizeof(int) * max_num_ctas));
    auto* num_non_exiting_ctas = static_cast<int*>(take(sizeof(int)));
    workspace.total_num_padded_tokens = total_num_padded_tokens;
    workspace.total_max_padded_tokens = max_num_padded_tokens;
    workspace.ProjUpTileN = tile;
    workspace.routing_expert_indexes = expert_indexes;
    workspace.permuted_idx_size = total_num_padded_tokens;
    workspace.expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx;
    workspace.permuted_idx_to_token_idx = permuted_idx_to_token_idx;
    workspace.permuted_idx_to_expanded_idx = nullptr;
    workspace.cta_idx_xy_to_batch_idx = cta_idx_xy_to_batch_idx;
    workspace.cta_idx_xy_to_mn_limit = cta_idx_xy_to_mn_limit;
    workspace.num_non_exiting_ctas = num_non_exiting_ctas;
    workspace.routing_expert_indexes = static_cast<int*>(a->topk_ids);
    workspace.expert_weights = a->topk_weights;
    args.mDtypeElt = dtype_act;
    btg::Dtype const routing_bias_dtype = btg::Dtype::Bfloat16, routing_logits_dtype = btg::Dtype::Bfloat16;
    args.mDtypeExpW = btg::Dtype::Fp32;  // UnpackedPrecomputed with float32 weights

    // FP4BlockScaleLauncher::run: routing (RoutingInputMode::UnpackedPrecomputed).
    Routing::Runner routing_runner(tile);
    routing_runner.run(args.routing_logits, args.routing_bias, t, e, k, args.num_fused_shared_experts, args.n_group,
                       args.topk_group, args.local_expert_offset, args.local_num_experts, args.routed_scaling_factor,
                       static_cast<int*>(a->topk_ids), expert_count_histogram, total_num_padded_tokens,
                       expanded_idx_to_permuted_idx, nullptr, permuted_idx_to_token_idx,
                       static_cast<int32_t*>(a->topk_ids), a->topk_weights, num_tokens_per_expert,
                       cta_idx_xy_to_batch_idx, cta_idx_xy_to_mn_limit, num_non_exiting_ctas, args.mDtypeElt,
                       routing_bias_dtype, /*useRoutingScalesOnInput=*/false, /*useDeepSeekFp8=*/false,
                       Routing::RoutingMethodType::Renormalize, stream, routing_logits_dtype,
                       /*normTopkProb=*/true, /*routingReplayOut=*/nullptr, a->enable_pdl != 0);

    // prepare_moe: configure_moe_args
    args.hidden_states = a->hidden_states;
    args.hidden_states_scale = a->hidden_states_scale;
    args.gemm1_weights = a->gemm1_weights;
    args.gemm1_weights_scale = a->gemm1_weights_scale;
    args.gemm1_alpha = nullptr;
    args.gemm1_beta = nullptr;
    args.gemm1_clamp_limit = a->gemm1_clamp_limit;
    args.gemm2_weights = a->gemm2_weights;
    args.gemm2_weights_scale = a->gemm2_weights_scale;
    args.gemm1_bias = nullptr;
    args.gemm2_bias = nullptr;
    args.output1_scales_scalar = a->output1_scales_scalar;
    args.output1_scales_gate_scalar = a->output1_scales_gate_scalar;
    args.output2_scales_scalar = a->output2_scales_scalar;

    // prepare_moe_runner: the runner and the tactic whose two GEMMs are the requested kernels.
    MoE::Runner runner(dtype_act, dtype_weights, /*useDeepSeekFp8=*/false, tile, MoE::ActivationType::Swiglu,
                       /*useShuffledMatrix=*/true, batchedGemm::gemm::MatrixLayout::MajorK,
                       batchedGemm::gemm::BiasType::None, false, false, false, false);
    int32_t const hidden_out = args.hidden_size;
    int64_t tactic = -1;
    std::string tried;
    for (int64_t c : runner.getValidConfigIndices(k, args.hidden_size, args.intermediate_size, args.local_num_experts, t,
                                                  hidden_out)) {
      auto parts = runner.getConfigComponents(c);
      std::string g1 = kernel_name(parts.gemm1Config), g2 = kernel_name(parts.gemm2Config);
      if (g1 == a->gemm1_kernel && g2 == a->gemm2_kernel) { tactic = c; break; }
      if (tried.size() < 4000) tried += "\n  " + g1 + " + " + g2;
    }
    if (tactic < 0) throw std::runtime_error(std::string("no valid MoE tactic uses ") + a->gemm1_kernel + " + " + a->gemm2_kernel + "; valid:" + tried);

    // prepare_moe_common + FP4BlockScaleLauncher::prepare_moe
    auto sizes = runner.getWorkspaceSizeInBytes(args, tactic);
    workspace.bmm1_workspace = take(std::get<0>(sizes));
    workspace.bmm2_workspace = take(std::get<1>(sizes));
    int const sf_vec_size = dtype_weights == btg::Dtype::MxE2m1 ? 32 : 16;
    int32_t const max_gemm1 = Routing::maybeGetMinTokenCount(workspace.total_max_padded_tokens, args.intermediate_size,
                                                             btg::dtypeGetNumBits(dtype_act));
    int32_t const max_gemm2 = Routing::maybeGetMinTokenCount(workspace.total_max_padded_tokens, hidden_out,
                                                             btg::dtypeGetNumBits(btg::Dtype::Bfloat16));
    int64_t const gemm1_output_hidden = dtype_act == btg::Dtype::E2m1 ? args.intermediate_size / 2 : args.intermediate_size;
    int64_t const sf_size = tensorrt_llm::computeSwizzledLayoutSFSize(max_gemm1, args.intermediate_size / sf_vec_size);
    void* gemm1_output_scale = take(sf_size);
    void* gemm1_output = take(static_cast<size_t>(max_gemm1) * gemm1_output_hidden);
    void* gemm2_output = take(static_cast<size_t>(max_gemm2) * hidden_out * 2);
    workspace.hidden_states_scale_linear = nullptr;
    workspace.gemm1_output = gemm1_output;
    workspace.gemm1_output_scale = static_cast<float*>(gemm1_output_scale);
    workspace.gemm2_output = gemm2_output;
    workspace.gemm2_output_scale = nullptr;
    if (a->scratch_used) *a->scratch_used = scratch.used;

    runner.run(args, workspace, device, stream, tactic, a->enable_pdl != 0);
    cudaError_t e2 = cudaGetLastError();
    if (e2 != cudaSuccess) throw std::runtime_error(std::string("MoE launch: ") + cudaGetErrorString(e2));
    return 0;
  } catch (std::exception const& ex) {
    std::snprintf(err, err_cap, "%s", ex.what());
    return 1;
  }
}

}  // extern "C"
