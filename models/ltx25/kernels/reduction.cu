// Pinned PyTorch cf30153c4c131c81 Reduce.cuh / TensorIterator host layout.
// Derived portions: BSD license in licenses/PYTORCH-LICENSE. Header-only kernels; no libtorch.
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/SharedReduceOps.h>
#include <c10/util/Exception.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// OffsetCalculator's TORCH_CHECK names libc10's failure hooks; nothing here links libc10 and the
// layouts built below never trip the checks.
namespace c10::detail {
[[noreturn]] void torchCheckFail(const char* func, const char* file, uint32_t line, const char* msg) {
    std::fprintf(stderr, "%s:%u %s: %s\n", file, unsigned(line), func, msg);
    std::abort();
}
[[noreturn]] void torchCheckFail(const char* func, const char* file, uint32_t line, const std::string& msg) {
    torchCheckFail(func, file, line, msg.c_str());
}
[[noreturn]] void torchInternalAssertFail(const char* func, const char* file, uint32_t line, const char* cond,
                                          const char* msg) {
    std::fprintf(stderr, "%s:%u %s: %s %s\n", file, unsigned(line), func, cond, msg);
    std::abort();
}
[[noreturn]] void torchInternalAssertFail(const char* func, const char* file, uint32_t line, const char* cond,
                                          const std::string& msg) {
    torchInternalAssertFail(func, file, line, cond, msg.c_str());
}
}  // namespace c10::detail

namespace {
using at::native::ReduceConfig;

constexpr int kMaxDims = 8;
constexpr int kThreads = 256;
inline unsigned blocks(int64_t n) { return unsigned((n + kThreads - 1) / kThreads); }

// ---- reductions ---------------------------------------------------------------------------------

// A TensorIterator reduction over one input and one (keepdim-viewed) output, after
// reorder_dimensions and coalesce_dimensions. Strides are bytes; reduced dims have output stride 0.
struct Layout {
    int ndim = 0;
    int64_t shape[kMaxDims];
    int64_t stride[2][kMaxDims];  // [0] output, [1] input

    int num_reduce_dims() const {
        int n = 0;
        for (int d = 0; d < ndim; ++d) n += stride[0][d] == 0;
        return n;
    }
    int64_t numel() const {
        int64_t n = 1;
        for (int d = 0; d < ndim; ++d) n *= shape[d];
        return n;
    }
    int64_t num_outputs() const {
        int64_t n = 1;
        for (int d = 0; d < ndim; ++d)
            if (stride[0][d] != 0 || shape[d] == 0) n *= shape[d];
        return n;
    }
};

// TensorIterator::reduce_op(result, self): result is freshly allocated (contiguous over the kept
// dims), viewed with size 1 / stride 0 at the reduced dims.
Layout reduction_layout(int ndim, const int64_t* sizes, const int64_t* strides, uint32_t reduce_mask, int64_t elem) {
    Layout l;
    l.ndim = ndim;
    int64_t out_stride = 1;
    for (int d = ndim - 1; d >= 0; --d) {
        l.shape[d] = sizes[d];
        l.stride[1][d] = strides[d] * elem;
        if (reduce_mask >> d & 1) {
            l.stride[0][d] = 0;
        } else {
            l.stride[0][d] = out_stride * elem;
            out_stride *= sizes[d];
        }
    }
    if (ndim > 1) {
        // reorder_dimensions: insertion sort, reduced dims first, then ascending strides.
        int perm[kMaxDims];
        for (int i = 0; i < ndim; ++i) perm[i] = ndim - 1 - i;
        auto should_swap = [&](int d0, int d1) {
            for (int arg = 0; arg < 2; ++arg) {
                const int64_t s0 = l.stride[arg][d0], s1 = l.stride[arg][d1];
                if (arg == 0 && ((s0 == 0) != (s1 == 0))) return s1 == 0 ? 1 : -1;
                if (s0 == 0 || s1 == 0) continue;
                if (s0 < s1) return -1;
                if (s0 > s1) return 1;
                if (l.shape[d0] > l.shape[d1]) return 1;
            }
            return 0;
        };
        for (int i = 1; i < ndim; ++i) {
            int dim1 = i;
            for (int dim0 = i - 1; dim0 >= 0; --dim0) {
                const int c = should_swap(perm[dim0], perm[dim1]);
                if (c > 0) {
                    std::swap(perm[dim0], perm[dim1]);
                    dim1 = dim0;
                } else if (c < 0) {
                    break;
                }
            }
        }
        Layout p = l;
        for (int i = 0; i < ndim; ++i) {
            p.shape[i] = l.shape[perm[i]];
            p.stride[0][i] = l.stride[0][perm[i]];
            p.stride[1][i] = l.stride[1][perm[i]];
        }
        l = p;
        // coalesce_dimensions
        auto can_coalesce = [&](int d0, int d1) {
            if (l.shape[d0] == 1 || l.shape[d1] == 1) return true;
            for (int arg = 0; arg < 2; ++arg)
                if (l.shape[d0] * l.stride[arg][d0] != l.stride[arg][d1]) return false;
            return true;
        };
        int prev = 0;
        for (int d = 1; d < ndim; ++d) {
            if (can_coalesce(prev, d)) {
                if (l.shape[prev] == 1)
                    for (int arg = 0; arg < 2; ++arg) l.stride[arg][prev] = l.stride[arg][d];
                l.shape[prev] *= l.shape[d];
            } else {
                ++prev;
                if (prev != d) {
                    for (int arg = 0; arg < 2; ++arg) l.stride[arg][prev] = l.stride[arg][d];
                    l.shape[prev] = l.shape[d];
                }
            }
        }
        l.ndim = prev + 1;
    }
    return l;
}

int last_pow2(int n) {
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    return std::max(1, n - (n >> 1));
}

int output_vec_size(const Layout& l, const void* input, int scalar_size) {
    int vec = 4;
    auto update = [&vec](uint64_t n) {
        while (n % vec != 0) vec /= 2;
    };
    update(reinterpret_cast<uint64_t>(input) / scalar_size);
    const int output_index = l.num_reduce_dims();
    update(l.shape[output_index]);
    for (int j = 0; j < l.ndim; ++j)
        if (j != output_index) update(l.stride[1][j] / scalar_size);
    return vec;
}

// setReduceConfig with PyTorch max_reduce_threads=512 and warp size 32.
ReduceConfig reduce_config(const Layout& l, int arg_size, int scalar_size, int input_vec_size, const void* input,
                           int num_mp, int max_threads_per_mp) {
    const int64_t num_outputs = l.num_outputs();
    const int64_t inputs_per_output = l.numel() / num_outputs;
    ReduceConfig config(arg_size, int(num_outputs), int(inputs_per_output));
    const int nrd = l.num_reduce_dims();
    int64_t dim0, dim1, fastest_moving_stride;
    bool on_fastest;
    if (l.ndim > 0) {
        on_fastest = nrd == l.ndim || l.stride[1][0] < l.stride[1][nrd];
        if (on_fastest) {
            dim0 = inputs_per_output;
            dim1 = num_outputs;
            fastest_moving_stride = l.stride[1][0];
        } else {
            dim0 = num_outputs;
            dim1 = inputs_per_output;
            fastest_moving_stride = l.stride[1][nrd];
        }
    } else {
        on_fastest = true;
        fastest_moving_stride = scalar_size;
        dim0 = dim1 = 1;
    }
    if (fastest_moving_stride == scalar_size) {
        if (on_fastest && dim0 >= 128 && nrd == 1) {
            config.vectorize_input = true;
            dim0 /= input_vec_size;
        } else if (!on_fastest) {
            config.output_vec_size = output_vec_size(l, input, scalar_size);
            dim0 /= config.output_vec_size;
        }
    }
    const int max_num_threads = 512 / config.output_vec_size;
    const int dim0_pow2 = dim0 < max_num_threads ? last_pow2(int(dim0)) : max_num_threads;
    const int dim1_pow2 = dim1 < max_num_threads ? last_pow2(int(dim1)) : max_num_threads;
    config.block_width = std::min(dim0_pow2, 32);
    config.block_height = std::min(dim1_pow2, max_num_threads / config.block_width);
    config.block_width = std::min(dim0_pow2, max_num_threads / config.block_height);
    config.num_threads = config.block_width * config.block_height;
    if (l.ndim == 0 || on_fastest) config.input_mult[0] = config.split_input(config.block_width);
    else config.output_mult[0] = config.split_output(config.block_width);
    constexpr int min_values_per_thread = 16, max_values_per_thread = 256;
    const int warp_split_threshold = std::min<int>(config.block_height * 16, max_values_per_thread);
    if (config.values_per_thread() >= warp_split_threshold) config.input_mult[1] = config.split_input(config.block_height);
    else config.output_mult[1] = config.split_output(config.block_height);
    const int blocks_per_sm = max_threads_per_mp / config.num_threads;
    const int target_grid_size = num_mp * blocks_per_sm;
    const int grid = config.grid().x;
    if (config.input_mult[1] != 0 && config.values_per_thread() >= max_values_per_thread && grid <= target_grid_size) {
        const int c1 = int(at::native::div_up(target_grid_size, grid));
        const int c2 = int(at::native::div_up(config.values_per_thread(), min_values_per_thread));
        const int c3 = int(at::native::div_up(config.values_per_thread(), max_values_per_thread));
        config.ctas_per_output = std::max(std::min<int>(c1, c2), c3);
        if (config.ctas_per_output > 1) config.input_mult[2] = config.split_input(config.ctas_per_output);
    }
    return config;
}

int shared_memory_size(const ReduceConfig& c) {
    if (!c.should_block_y_reduce() && (!c.should_block_x_reduce() || c.block_width <= 32)) return 0;
    return c.element_size_bytes * c.num_threads * c.output_vec_size;
}

// gpu_reduce_kernel for a 32-bit-indexable iterator: one launch, no accumulation buffer.
template <typename scalar_t, typename out_t, int vt0, int input_vec_size, typename ops_t, typename arg_t>
int launch_reduce(const Layout& l, const ops_t& ops, arg_t ident, const void* in, void* out, cudaStream_t stream) {
    int device = 0, num_mp = 0, max_threads = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_mp, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, device);
    const ReduceConfig config = reduce_config(l, sizeof(arg_t), sizeof(scalar_t), input_vec_size, in, num_mp, max_threads);
    void* buffer = nullptr;
    int* semaphores = nullptr;
    if (config.should_global_reduce()) {
        if (auto e = cudaMallocAsync(&buffer, config.global_memory_size(), stream)) return e;
        if (auto e = cudaMallocAsync(reinterpret_cast<void**>(&semaphores), config.semaphore_size(), stream)) return e;
        if (auto e = cudaMemsetAsync(semaphores, 0, config.semaphore_size(), stream)) return e;
    }
    const int nrd = l.num_reduce_dims();
    const int64_t* out_strides[2] = {l.stride[0] + nrd, l.stride[1] + nrd};
    ::OffsetCalculator<2, uint32_t> output_calc(l.ndim - nrd, l.shape + nrd, out_strides);
    const int64_t* in_strides[1] = {l.stride[1]};
    ::OffsetCalculator<1, uint32_t> input_calc(nrd, l.shape, in_strides);
    using R = at::native::ReduceOp<scalar_t, ops_t, uint32_t, out_t, vt0, input_vec_size>;
    R reduce(ops, config, input_calc, output_calc, in, static_cast<char*>(out), std::nullopt, nullptr, buffer, semaphores,
             ident, 1, 0);
    reduce.accumulate = false;
    reduce.final_output = true;
    const int smem = shared_memory_size(config);
    switch (config.output_vec_size) {
    case 4: at::native::reduce_kernel<512 / 4, 4, R><<<config.grid(), config.block(), smem, stream>>>(reduce); break;
    case 2: at::native::reduce_kernel<512 / 2, 2, R><<<config.grid(), config.block(), smem, stream>>>(reduce); break;
    default: at::native::reduce_kernel<512, 1, R><<<config.grid(), config.block(), smem, stream>>>(reduce);
    }
    int err = cudaGetLastError();
    if (buffer) {
        cudaFreeAsync(buffer, stream);
        cudaFreeAsync(semaphores, stream);
    }
    return err;
}

} // namespace
extern "C" int ltx25_mean(const void* input, void* output, int dtype, int ndim,
                           const int64_t* sizes, const int64_t* strides, uint32_t mask, void* stream) {
    if (ndim < 1 || ndim > kMaxDims || mask == 0 || (dtype != 0 && dtype != 1)) return -1;
    auto l = reduction_layout(ndim, sizes, strides, mask, dtype == 0 ? 2 : 4);
    if (l.numel() <= 0 || l.numel() > INT32_MAX || l.num_outputs() <= 0) return -2;
    const float factor = static_cast<float>(l.num_outputs()) / l.numel();
    auto st = static_cast<cudaStream_t>(stream);
    if (dtype == 0) {
        using T = at::BFloat16;
        using Ops = at::native::MeanOps<T, float, float, T>;
        return launch_reduce<T, T, 4, 8>(l, Ops{factor}, 0.0f, input, output, st);
    }
    using Ops = at::native::MeanOps<float, float, float, float>;
    return launch_reduce<float, float, 4, 4>(l, Ops{factor}, 0.0f, input, output, st);
}
extern "C" int ltx25_norm2(const void* input, void* output, int ndim,
                            const int64_t* sizes, const int64_t* strides, uint32_t mask, void* stream) {
    if (ndim < 1 || ndim > kMaxDims || !mask) return -1;
    auto l = reduction_layout(ndim, sizes, strides, mask, 4);
    if (l.numel() <= 0 || l.numel() > INT32_MAX || l.num_outputs() <= 0) return -2;
    using Ops = at::native::NormTwoOps<float, float, float>;
    return launch_reduce<float,float,4,4>(l, Ops{}, 0.0f, input, output, static_cast<cudaStream_t>(stream));
}
