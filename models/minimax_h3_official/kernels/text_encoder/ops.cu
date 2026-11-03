// Qwen3-VL-specific native arithmetic; CUDA only, no ATen/Python headers.
// Reduction, LayerNorm and GELU order follows pinned PyTorch 2.13.0.
// Derived reduction/Welford portions retain reference_sources/LICENSE.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

using B = __nv_bfloat16;
namespace {
__device__ float f(B x) { return __bfloat162float(x); }
__device__ B b(float x) { return __float2bfloat16_rn(x); }
__device__ float roundb(float x) { return f(b(x)); }
unsigned grid(int64_t n) { return unsigned((n + 255) / 256 > 65535 ? 65535 : (n + 255) / 256); }

// Qwen's RMSNorm is x.float().pow(2).mean, then float normalization,
// BF16 materialization, then BF16 weight multiplication. It is not torch.rms_norm.
// For contiguous widths 128/5120 and these row counts, Reduce.cuh selects
// vectorized four-lane accumulators and one 32-lane warp per row.
__global__ void rms(const B* x, const B* weight, B* out, int64_t rows, int cols, float eps) {
    int lane = threadIdx.x;
    int64_t row = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    float sum[4] = {0, 0, 0, 0};
    for (int vector = lane; vector < cols / 4; vector += 32) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float v = f(x[row * cols + vector * 4 + j]);
            sum[j] = __fadd_rn(sum[j], __fmul_rn(v, v));
        }
    }
    float value = __fadd_rn(__fadd_rn(__fadd_rn(sum[0], sum[1]), sum[2]), sum[3]);
    for (int offset = 16; offset; offset >>= 1)
        value = __fadd_rn(value, __shfl_down_sync(0xffffffff, value, offset));
    float mean = __fmul_rn(__shfl_sync(0xffffffff, value, 0), 1.f / float(cols));
    float rstd = rsqrtf(__fadd_rn(mean, eps));
    for (int col = lane; col < cols; col += 32)
        out[row * cols + col] = b(__fmul_rn(f(weight[col]), roundb(__fmul_rn(f(x[row * cols + col]), rstd))));
}

struct W { float mean, sigma2, count; };
__device__ W online(float val, W a) {
    float delta = val - a.mean;
    float count = a.count + 1.f;
    float mean = a.mean + delta * (1.f / count);
    return {mean, a.sigma2 + delta * (val - mean), count};
}
__device__ W combine(W b, W a) {
    float delta = b.mean - a.mean;
    float count = a.count + b.count;
    if (count <= 0) return {0, 0, count};
    float coef = 1.f / count, na = a.count * coef, nb = b.count * coef;
    return {na * a.mean + nb * b.mean, a.sigma2 + b.sigma2 + delta * delta * a.count * nb, count};
}
__global__ void layernorm(const B* x, const B* weight, const B* bias, B* out, int cols, float eps) {
    __shared__ float smem[8];
    int lane = threadIdx.x, warp = threadIdx.y, tid = lane + 32 * warp;
    int64_t offset = int64_t(blockIdx.x) * cols;
    W wd{0, 0, 0};
    for (int vector = tid; vector < cols / 4; vector += 128) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) wd = online(f(x[offset + vector * 4 + j]), wd);
    }
    for (int delta = 16; delta; delta >>= 1) {
        W other{__shfl_down_sync(0xffffffff, wd.mean, delta),
                __shfl_down_sync(0xffffffff, wd.sigma2, delta),
                __shfl_down_sync(0xffffffff, wd.count, delta)};
        wd = combine(wd, other);
    }
    for (int delta = 2; delta; delta >>= 1) {
        if (!lane && warp >= delta && warp < 2 * delta) {
            smem[2 * (warp - delta)] = wd.mean;
            smem[2 * (warp - delta) + 1] = wd.sigma2;
            smem[4 + warp - delta] = wd.count;
        }
        __syncthreads();
        if (!lane && warp < delta) wd = combine(wd, {smem[2 * warp], smem[2 * warp + 1], smem[4 + warp]});
        __syncthreads();
    }
    if (!lane && !warp) { smem[0] = wd.mean; smem[1] = wd.sigma2 / float(cols); }
    __syncthreads();
    float rstd = rsqrtf(smem[1] + eps);
    for (int col = tid; col < cols; col += 128)
        out[offset + col] = b(f(weight[col]) * (rstd * (f(x[offset + col]) - smem[0])) + f(bias[col]));
}

__global__ void activation(const B* x, B* out, int64_t n, int kind) {
    for (int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += int64_t(gridDim.x) * 256) {
        float v = f(x[i]);
        if (kind == 0) out[i] = b(v / (1.f + expf(-v)));
        else if (kind == 1) {
            constexpr float beta = 0.79788456080286535587989211986876;
            constexpr float kappa = 0.044715;
            float cube = v * v * v;
            float inner = beta * (v + kappa * cube);
            out[i] = b(0.5f * v * (1.f + tanhf(inner)));
        } else out[i] = b(v * 0.5f * (1.f + erff(v * 0.70710678118654752440084436210485f)));
    }
}
__global__ void binary(const B* x, const B* y, B* out, int64_t n, int kind, int ycols) {
    for (int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += int64_t(gridDim.x) * 256) {
        float a = f(x[i]), c = f(y[ycols ? i % ycols : i]);
        if (kind == 0) out[i] = b(a + c);
        else if (kind == 1) out[i] = b(a * c);
        else out[i] = b(roundb(a / (1.f + expf(-a))) * c);
    }
}
__global__ void cast_bf16(const float* x, B* out, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += int64_t(gridDim.x) * 256) out[i] = b(x[i]);
}
__global__ void embedding(const int64_t* ids, const B* table, B* out, int64_t n, int width) {
    for (int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += int64_t(gridDim.x) * 256)
        out[i] = table[ids[i / width] * width + i % width];
}
__global__ void vision_rope(const B* qkv, const float* cos, const float* sin, B* q, B* k,
                            int64_t n, int inner, int dim) {
    for (int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += int64_t(gridDim.x) * 256) {
        int d = i % dim, half = dim / 2;
        int64_t src = (i / inner) * (3 * inner) + i % inner;
        int64_t partner = src - d + (d < half ? d + half : d - half);
        int64_t pos = (i / inner) * dim + d;
        float sign = d < half ? -1.f : 1.f;
        q[i] = b(__fadd_rn(__fmul_rn(f(qkv[src]), cos[pos]), __fmul_rn(sign * f(qkv[partner]), sin[pos])));
        k[i] = b(__fadd_rn(__fmul_rn(f(qkv[src + inner]), cos[pos]), __fmul_rn(sign * f(qkv[partner + inner]), sin[pos])));
    }
}
__global__ void text_rope(const B* x, const B* cos, const B* sin, B* out, int64_t n, int heads) {
    for (int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += int64_t(gridDim.x) * 256) {
        int d = i % 128;
        int64_t partner = i - d + (d < 64 ? d + 64 : d - 64);
        int64_t pos = (i / (heads * 128)) * 128 + d;
        float rotated = f(x[partner]); if (d < 64) rotated = -rotated;
        out[i] = b(roundb(f(x[i]) * f(cos[pos])) + roundb(rotated * f(sin[pos])));
    }
}
// Match CUDA linspace: compute a rounded step, then use the nearest endpoint
// (and a fused operation for the second half). This rounding also matters at
// BF16 precision after interpolation.
__device__ float vision_coordinate(int i, int n) {
    float step=__fdiv_rn(47.f,float(n-1));
    return i<n/2 ? __fmul_rn(step,float(i)) : fmaf(-step,float(n-1-i),47.f);
}
// Align-corners bilinear interpolation and merged-grid order follow the
// SGLang Qwen vision encoder. Preserve FP32 weights and the four-term sum
// before converting the position embedding to BF16.
__global__ void vision_tables(const B* pos_table,const float* inv,float* cos,float* sin,B* position,int gh,int gw) {
    int rows=2*gh*gw;
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*1152;i+=gridDim.x*blockDim.x) {
        int token=i/1152,channel=i%1152,local=token%(gh*gw);
        int y=(local/(2*gw))*2+(local/2)%2,x=((local/4)%(gw/2))*2+local%2;
        if (gh==48 && gw==48) {
            position[i]=pos_table[(y*48+x)*1152+channel];
        } else {
            float sy=vision_coordinate(y,gh),sx=vision_coordinate(x,gw);
            int y0=int(floorf(sy)),x0=int(floorf(sx));
            float fy=__fsub_rn(sy,float(y0)),fx=__fsub_rn(sx,float(x0));
            float wy[2]={__fsub_rn(1.f,fy),fy};
            float wx[2]={__fsub_rn(1.f,fx),fx};
            float sum=0.f;
            for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx) {
                int ty=min(y0+dy,47),tx=min(x0+dx,47);
                float weight=__fmul_rn(wy[dy],wx[dx]);
                sum=__fadd_rn(sum,__fmul_rn(f(pos_table[(ty*48+tx)*1152+channel]),weight));
            }
            position[i]=b(sum);
        }
    }
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*72;i+=gridDim.x*blockDim.x) {
        int token=i/72,channel=(i%72)%36,local=token%(gh*gw);
        int y=(local/(2*gw))*2+(local/2)%2,x=((local/4)%(gw/2))*2+local%2;
        float value=__fmul_rn(float(channel<18?y:x),inv[channel%18]);
        cos[i]=cosf(value);sin[i]=sinf(value);
    }
}
__global__ void text_tables(const int64_t* positions,const float* inv,B* cos,B* sin,int rows) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*128;i+=gridDim.x*blockDim.x) {
        int token=i/128,channel=i%64,axis=0;
        if (channel<60 && channel%3==1) axis=1;
        else if (channel<60 && channel%3==2) axis=2;
        float value=__fmul_rn(float(positions[axis*rows+token]),inv[channel]);
        cos[i]=b(cosf(value));sin[i]=b(sinf(value));
    }
}
__global__ void image_insert(B* hidden,const B* images,const int64_t* index,int64_t n,int add) {
    for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x) {
        int64_t target=index[i/5120]*5120+i%5120;
        hidden[target]=add?b(f(hidden[target])+f(images[i])):images[i];
    }
}
}
extern "C" int q3_rms(const void* x, const void* w, void* out, int64_t rows, int cols, float eps, cudaStream_t stream) {
    // A short text-only prompt can have fewer than sixteen rows. Each warp
    // owns one row; the kernel already skips inactive warps in the final block.
    if (rows < 1 || (cols != 128 && cols != 5120)) return cudaErrorInvalidValue;
    rms<<<unsigned((rows + 15) / 16), dim3(32,16), 0, stream>>>((const B*)x,(const B*)w,(B*)out,rows,cols,eps); return cudaGetLastError();
}
extern "C" int q3_layernorm(const void* x, const void* w, const void* bias, void* out, int64_t rows, int cols, float eps, cudaStream_t stream) {
    if (rows < 1 || (cols != 1152 && cols != 4608)) return cudaErrorInvalidValue;
    layernorm<<<unsigned(rows),dim3(32,4),0,stream>>>((const B*)x,(const B*)w,(const B*)bias,(B*)out,cols,eps); return cudaGetLastError();
}
extern "C" int q3_activation(const void* x, void* out, int64_t n, int kind, cudaStream_t stream) {
    if (kind < 0 || kind > 2 || n < 1) return cudaErrorInvalidValue;
    activation<<<grid(n),256,0,stream>>>((const B*)x,(B*)out,n,kind); return cudaGetLastError();
}
extern "C" int q3_binary(const void* x, const void* y, void* out, int64_t n, int kind, int ycols, cudaStream_t stream) {
    binary<<<grid(n),256,0,stream>>>((const B*)x,(const B*)y,(B*)out,n,kind,ycols); return cudaGetLastError();
}
extern "C" int q3_cast_bf16(const float* x, void* out, int64_t n, cudaStream_t stream) {
    cast_bf16<<<grid(n),256,0,stream>>>(x,(B*)out,n); return cudaGetLastError();
}
extern "C" int q3_embedding(const int64_t* ids, const void* table, void* out, int64_t rows, int width, cudaStream_t stream) {
    embedding<<<grid(rows*width),256,0,stream>>>(ids,(const B*)table,(B*)out,rows*width,width); return cudaGetLastError();
}
extern "C" int q3_vision_rope(const void* qkv, const float* cos, const float* sin, void* q, void* k, int64_t rows, cudaStream_t stream) {
    vision_rope<<<grid(rows*1152),256,0,stream>>>((const B*)qkv,cos,sin,(B*)q,(B*)k,rows*1152,1152,72); return cudaGetLastError();
}
extern "C" int q3_text_rope(const void* x, const void* cos, const void* sin, void* out, int64_t rows, int heads, cudaStream_t stream) {
    text_rope<<<grid(rows*heads*128),256,0,stream>>>((const B*)x,(const B*)cos,(const B*)sin,(B*)out,rows*heads*128,heads); return cudaGetLastError();
}
extern "C" int q3_vision_tables(const void* pos_table,const float* inv,float* cos,float* sin,void* position,int gh,int gw,cudaStream_t stream) {
    if (gh<2 || gw<2 || gh%2 || gw%2 || int64_t(gh)*gw>1048576) return cudaErrorInvalidValue;
    vision_tables<<<grid(int64_t(2)*gh*gw*1152),256,0,stream>>>((const B*)pos_table,inv,cos,sin,(B*)position,gh,gw);return cudaGetLastError();
}
extern "C" int q3_text_tables(const int64_t* position,const float* inv,void* cos,void* sin,int rows,cudaStream_t stream) {
    if (rows<1) return cudaErrorInvalidValue;
    text_tables<<<grid(int64_t(rows)*128),256,0,stream>>>(position,inv,(B*)cos,(B*)sin,rows);return cudaGetLastError();
}
extern "C" int q3_image_insert(void* hidden,const void* images,const int64_t* index,int image_tokens,int add,cudaStream_t stream) {
    image_insert<<<grid(int64_t(image_tokens)*5120),256,0,stream>>>((B*)hidden,(const B*)images,index,int64_t(image_tokens)*5120,add);return cudaGetLastError();
}
