"""Probe: compile one flashinfer MoE source with the native build's flags (diagnosing compiler crashes)."""
import subprocess, sys, time
fi, mb, cuda = '/dsv4/deps/flashinfer', '/dsv4/deps/moe_build', '/usr/local/cuda'
csrc = fi + '/csrc'
common = ['-D_GLIBCXX_USE_CXX11_ABI=1'] + ['-I' + d for d in [mb + '/gen_root_fp4', csrc + '/nv_internal', csrc + '/nv_internal/include',
          mb + '/data/cccl/cub', mb + '/data/cccl/libcudacxx/include', mb + '/data/cccl/thrust']]
for d in [cuda + '/include', mb + '/tvm_ffi/include', fi + '/include', csrc, mb + '/data/cutlass/include',
          mb + '/data/cutlass/tools/util/include', mb + '/data/spdlog/include']:
    common += ['-isystem', d]
flags = ['--compiler-options=-fPIC', '--expt-relaxed-constexpr', '-static-global-template-stub=false', '-std=c++17', '--threads=1',
         '-use_fast_math', '-DNDEBUG', '-O3', '-DTLLM_GEN_EXPORT_INTERFACE', '-DTLLM_GEN_EXPORT_FLASHINFER', '-DTLLM_ENABLE_CUDA',
         '-DENABLE_BF16', '-DENABLE_FP8', '-DENABLE_FP4', '-DCUTLASS_ENABLE_GDC_FOR_SM100=1',
         '-DTLLM_GEN_GEMM_CUBIN_PATH="35cb99413a3ebce2e87a03fc04196e528e22c8d5/batched_gemm-fdba669-7e3d0a6/"',
         '-gencode=arch=compute_103a,code=sm_103a']
src = sys.argv[1] if len(sys.argv) > 1 else csrc + '/trtllm_batched_gemm_runner.cu'
for c in ['gcc --version | head -1', 'nvcc --version | tail -2', 'ulimit -s; ulimit -Hs', 'free -g | head -2']:
    print(c, '=>', subprocess.run(['bash', '-lc', c], capture_output=True, text=True).stdout.strip())
import resource
def unlimited():
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
for extra, pre in [([], unlimited)]:
    t = time.time()
    r = subprocess.run(['nvcc'] + common + flags + extra + ['-c', src, '-o', '/tmp/probe.o'], capture_output=True, text=True, preexec_fn=pre)
    print(f'== extra={extra} rc={r.returncode} {time.time()-t:.0f}s')
    print('\n'.join(l[:400] for l in r.stderr.splitlines()[-12:]))
    if r.returncode == 0:
        break
