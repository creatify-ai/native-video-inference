// Kernel-launch redirection for reference host code compiled into the native binary (e.g.
// flashinfer's trtllm MoE runner). That code builds its parameter structs exactly as the reference
// does and launches its own freshly compiled kernels; these definitions of the CUDA runtime launch
// entry points (which take precedence over libcudart's for this binary's objects) hand every such
// launch to the registered handler, which launches the reference's CUBIN of the same name with
// the same configuration and parameter bytes. Without a handler, or when the handler declines,
// the launch goes to the real runtime.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>

extern "C" {

// Handler: returns 1 if it launched (or failed: *err set), 0 to fall through to the real runtime.
// attrs/num_attrs mirror cudaLaunchAttribute (same layout as CUlaunchAttribute).
typedef int (*dsv4_launch_handler)(const char* name, const unsigned grid[3], const unsigned block[3], unsigned smem,
                                   void* stream, void** args, const void* attrs, unsigned num_attrs, int* err);

static dsv4_launch_handler g_handler = nullptr;

__attribute__((visibility("default"))) void dsv4_set_launch_handler(dsv4_launch_handler h) { g_handler = h; }

// Parameter layout of the function behind a runtime kernel symbol (for the handler to size args).
__attribute__((visibility("default"))) int dsv4_symbol_name(const void* func, const char** name) {
    if (cudaFuncGetName(name, func) == cudaSuccess && *name) return 1;
    cudaFunction_t f = nullptr;
    if (cudaGetFuncBySymbol(&f, func) != cudaSuccess || !f) return 0;
    return cuFuncGetName(name, (CUfunction)f) == CUDA_SUCCESS;
}

typedef cudaError_t (*launch_fn)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
typedef cudaError_t (*launch_ex_fn)(const cudaLaunchConfig_t*, const void*, void**);

static bool log_misses() { static int v = getenv("DSV4_REDIRECT_LOG") ? 1 : 0; return v; }

cudaError_t cudaLaunchKernel(const void* func, dim3 grid, dim3 block, void** args, size_t smem, cudaStream_t stream) {
    const char* name = nullptr;
    if (log_misses()) fprintf(stderr, "cudaLaunchKernel intercepted (handler %s)\n", g_handler ? "set" : "unset");
    if (g_handler && dsv4_symbol_name(func, &name)) {
        unsigned g[3] = {grid.x, grid.y, grid.z}, b[3] = {block.x, block.y, block.z};
        int err = 0;
        if (g_handler(name, g, b, (unsigned)smem, stream, args, nullptr, 0, &err)) return err ? cudaErrorLaunchFailure : cudaSuccess;
    }
    static launch_fn real = (launch_fn)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    return real(func, grid, block, args, smem, stream);
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func, void** args) {
    const char* name = nullptr;
    if (log_misses()) fprintf(stderr, "cudaLaunchKernelExC intercepted (handler %s)\n", g_handler ? "set" : "unset");
    if (g_handler && dsv4_symbol_name(func, &name)) {
        unsigned g[3] = {config->gridDim.x, config->gridDim.y, config->gridDim.z};
        unsigned b[3] = {config->blockDim.x, config->blockDim.y, config->blockDim.z};
        int err = 0;
        if (g_handler(name, g, b, (unsigned)config->dynamicSmemBytes, config->stream, args, config->attrs, config->numAttrs, &err))
            return err ? cudaErrorLaunchFailure : cudaSuccess;
    }
    static launch_ex_fn real = (launch_ex_fn)dlsym(RTLD_NEXT, "cudaLaunchKernelExC");
    return real(config, func, args);
}

// CUDA 13's <<<>>> stubs launch through this internal entry with a cudaKernel_t handle (see
// crt/host_runtime.h __cudaLaunch / crt/device_functions.h __cudaLaunchKernel_helper).
typedef cudaError_t (*launch_kernel_fn)(cudaKernel_t, dim3, dim3, void**, size_t, cudaStream_t);

static int redirect_kernel_handle(cudaKernel_t kernel, dim3 grid, dim3 block, void** args, size_t smem, cudaStream_t stream,
                                  cudaError_t* out) {
    const char* name = nullptr;
    if (!g_handler || cuKernelGetName(&name, (CUkernel)kernel) != CUDA_SUCCESS || !name) return 0;
    unsigned g[3] = {grid.x, grid.y, grid.z}, b[3] = {block.x, block.y, block.z};
    int err = 0;
    if (!g_handler(name, g, b, (unsigned)smem, stream, args, nullptr, 0, &err)) return 0;
    *out = err ? cudaErrorLaunchFailure : cudaSuccess;
    return 1;
}

cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block, void** args, size_t smem, cudaStream_t stream) {
    if (log_misses()) fprintf(stderr, "__cudaLaunchKernel intercepted (handler %s)\n", g_handler ? "set" : "unset");
    cudaError_t r;
    if (redirect_kernel_handle(kernel, grid, block, args, smem, stream, &r)) return r;
    static launch_kernel_fn real = (launch_kernel_fn)dlsym(RTLD_NEXT, "__cudaLaunchKernel");
    return real(kernel, grid, block, args, smem, stream);
}

cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 grid, dim3 block, void** args, size_t smem, cudaStream_t stream) {
    cudaError_t r;
    if (redirect_kernel_handle(kernel, grid, block, args, smem, stream, &r)) return r;
    static launch_kernel_fn real = (launch_kernel_fn)dlsym(RTLD_NEXT, "__cudaLaunchKernel_ptsz");
    return real(kernel, grid, block, args, smem, stream);
}

}  // extern "C"
