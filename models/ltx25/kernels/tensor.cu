// Strided eager primitives. RMS reduction follows PyTorch cf30153c4c131c81,
// aten/native/cuda/layer_norm_kernel.cu (BSD license in licenses/PYTORCH-LICENSE).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <fstream>
#include <functional>
#include <sstream>
#include <unistd.h>
#include <nvrtc.h>

struct LtxView { void* data; int dtype, rank; int64_t dims[8], strides[8]; };
struct Pads { int64_t before[8]; };
// One op of a fused elementwise run (graph.rs): op code as ltx25_tensor_elementwise's, or 50 for a
// dtype cast; operands a, b, c are registers; round is the op's eager output dtype (0 bf16, 1 fp32).
struct LtxFuseOp { int op, a, b, c, round; float scalar, alpha; int scalar_b; };
namespace {
__device__ int64_t address(const LtxView& v, int64_t linear, const LtxView& output) {
    int64_t offset=0;
    for (int d=output.rank-1; d>=0; --d) {
        int64_t c=linear%output.dims[d]; linear/=output.dims[d];
        int j=d-(output.rank-v.rank);
        if (j>=0 && v.dims[j]!=1) offset+=c*v.strides[j];
    }
    return offset;
}
__device__ float read(const LtxView& v, int64_t i) {
    if (v.dtype==0) return __bfloat162float(((const __nv_bfloat16*)v.data)[i]);
    if (v.dtype==1) return ((const float*)v.data)[i];
    if (v.dtype==2) return float(((const int64_t*)v.data)[i]);
    if (v.dtype==4) return float(((const int32_t*)v.data)[i]);
    if (v.dtype==5) return __half2float(((const __half*)v.data)[i]);
    return float(((const uint8_t*)v.data)[i]);
}
__device__ void write(const LtxView& v, int64_t i, float x) {
    if (v.dtype==0) ((__nv_bfloat16*)v.data)[i]=__float2bfloat16_rn(x);
    else if(v.dtype==1) ((float*)v.data)[i]=x;
    else if(v.dtype==2) ((int64_t*)v.data)[i]=int64_t(x);
    else if(v.dtype==4) ((int32_t*)v.data)[i]=int32_t(x);
    else if(v.dtype==5) ((__half*)v.data)[i]=__float2half_rn(x);
    else ((uint8_t*)v.data)[i]=uint8_t(x);
}
// A launch's index plan: the indexed dims without size-1 dims, adjacent dims merged where every
// operand is contiguous across them, and each operand's element stride per merged dim (0 where it
// broadcasts). offsets() gives address()'s sums with fewer terms (one decomposition of the linear
// index for all operands, 32-bit when it fits); the elements and their values are unchanged.
struct Plan { int rank; int64_t dims[8]; int64_t st[4][8]; int shift[8]; };  // shift: log2(dims) or -1
Plan make_plan(const LtxView& by, const LtxView* const* views, int count) {
    int r = 0;
    int64_t dims[8], st[4][8];
    for (int d = 0; d < by.rank; ++d) {
        if (by.dims[d] == 1) continue;
        dims[r] = by.dims[d];
        for (int k = 0; k < count; ++k) {
            const LtxView& v = *views[k];
            const int j = d - (by.rank - v.rank);
            st[k][r] = j >= 0 && v.dims[j] != 1 ? v.strides[j] : 0;
        }
        ++r;
    }
    Plan p{};
    for (int d = 0; d < r; ++d) {
        bool merge = p.rank > 0;
        for (int k = 0; merge && k < count; ++k) merge = p.st[k][p.rank - 1] == st[k][d] * dims[d];
        if (merge) {
            p.dims[p.rank - 1] *= dims[d];
            for (int k = 0; k < count; ++k) p.st[k][p.rank - 1] = st[k][d];
        } else {
            p.dims[p.rank] = dims[d];
            for (int k = 0; k < count; ++k) p.st[k][p.rank] = st[k][d];
            ++p.rank;
        }
    }
    for (int d = 0; d < p.rank; ++d) {
        p.shift[d] = -1;
        if ((p.dims[d] & (p.dims[d] - 1)) == 0)
            for (int b = 0; b < 63; ++b) if ((int64_t(1) << b) == p.dims[d]) p.shift[d] = b;
    }
    return p;
}
template <typename I, int N>
__device__ __forceinline__ void offsets(const Plan& p, int64_t linear, int64_t* off) {
    #pragma unroll
    for (int k = 0; k < N; ++k) off[k] = 0;
    I rest = I(linear);
    for (int d = p.rank - 1; d > 0; --d) {
        const I extent = I(p.dims[d]);
        I c;
        if (p.shift[d] >= 0) { c = rest & (extent - 1); rest >>= p.shift[d]; }
        else { c = rest % extent; rest /= extent; }
        #pragma unroll
        for (int k = 0; k < N; ++k) off[k] += int64_t(c) * p.st[k][d];
    }
    if (p.rank > 0) {
        #pragma unroll
        for (int k = 0; k < N; ++k) off[k] += int64_t(rest) * p.st[k][0];
    }
}
inline unsigned grid_for(int64_t n) { return unsigned(std::min<int64_t>((n + 255) / 256, int64_t(1) << 20)); }
// Grouped forms: `group` consecutive output elements per thread, for launches whose innermost plan
// dim holds whole groups and whose operands are each unit-stride across it (16-byte accesses, each
// group's start aligned) or broadcast across it (one value per group). The group's elements are the
// same elements with the same per-element expressions; only the access width differs.
constexpr unsigned kNoGroups = ~0u;
// Bit k set when operand k is unit-stride across the inner dim; kNoGroups when the plan has no
// grouped form. Operands that `used` excludes index as zero-stride and impose nothing.
unsigned group_units(const Plan& p, const LtxView* const* views, const bool* used, int count, int group) {
    if (p.rank == 0 || p.dims[p.rank - 1] % group) return kNoGroups;
    unsigned unit = 0;
    for (int k = 0; k < count; ++k) {
        if (!used[k]) continue;
        const int64_t inner = p.st[k][p.rank - 1];
        if (inner == 0) continue;
        if (inner != 1 || reinterpret_cast<uintptr_t>(views[k]->data) % 16) return kNoGroups;
        for (int d = 0; d < p.rank - 1; ++d) if (p.st[k][d] % group) return kNoGroups;
        unit |= 1u << k;
    }
    return unit;
}
constexpr int kGroup = 8;
template <typename T> __device__ __forceinline__ void ld8(const void* p, int64_t i, bool unit, float* v);
template <> __device__ __forceinline__ void ld8<__nv_bfloat16>(const void* p, int64_t i, bool unit, float* v) {
    const __nv_bfloat16* x = (const __nv_bfloat16*)p + i;
    if (unit) {
        const uint4 r = *(const uint4*)x;
        const __nv_bfloat16* h = (const __nv_bfloat16*)&r;
        #pragma unroll
        for (int j = 0; j < kGroup; ++j) v[j] = __bfloat162float(h[j]);
    } else {
        const float s = __bfloat162float(*x);
        #pragma unroll
        for (int j = 0; j < kGroup; ++j) v[j] = s;
    }
}
template <> __device__ __forceinline__ void ld8<float>(const void* p, int64_t i, bool unit, float* v) {
    const float* x = (const float*)p + i;
    if (unit) {
        const float4 r0 = ((const float4*)x)[0], r1 = ((const float4*)x)[1];
        v[0] = r0.x; v[1] = r0.y; v[2] = r0.z; v[3] = r0.w; v[4] = r1.x; v[5] = r1.y; v[6] = r1.z; v[7] = r1.w;
    } else {
        const float s = *x;
        #pragma unroll
        for (int j = 0; j < kGroup; ++j) v[j] = s;
    }
}
template <> __device__ __forceinline__ void ld8<uint8_t>(const void* p, int64_t i, bool unit, float* v) {
    const uint8_t* x = (const uint8_t*)p + i;
    if (unit) {
        const uint2 r = *(const uint2*)x;
        const uint8_t* b = (const uint8_t*)&r;
        #pragma unroll
        for (int j = 0; j < kGroup; ++j) v[j] = float(b[j]);
    } else {
        const float s = float(*x);
        #pragma unroll
        for (int j = 0; j < kGroup; ++j) v[j] = s;
    }
}
template <typename T> __device__ __forceinline__ void st8(void* p, int64_t i, const float* z);
template <> __device__ __forceinline__ void st8<__nv_bfloat16>(void* p, int64_t i, const float* z) {
    uint4 r;
    __nv_bfloat16* h = (__nv_bfloat16*)&r;
    #pragma unroll
    for (int j = 0; j < kGroup; ++j) h[j] = __float2bfloat16_rn(z[j]);
    *(uint4*)((__nv_bfloat16*)p + i) = r;
}
template <> __device__ __forceinline__ void st8<float>(void* p, int64_t i, const float* z) {
    float4* y = (float4*)((float*)p + i);
    y[0] = make_float4(z[0], z[1], z[2], z[3]);
    y[1] = make_float4(z[4], z[5], z[6], z[7]);
}
// Same-dtype copy: 16 bytes (16/width elements) per thread.
__global__ void copy_chunks(const uint8_t* x, uint8_t* y, Plan px, Plan py, int64_t chunks, int per, int width) {
    for (int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; g < chunks; g += int64_t(gridDim.x) * blockDim.x) {
        int64_t xi, yi;
        offsets<uint32_t, 1>(px, g * per, &xi);
        offsets<uint32_t, 1>(py, g * per, &yi);
        *(uint4*)(y + yi * width) = *(const uint4*)(x + xi * width);
    }
}
// bf16 <-> fp32 conversion copy, eight elements per thread (write(read(x)) per element).
template <typename TX, typename TY>
__global__ void convert_group(const void* x, void* y, Plan px, Plan py, int64_t groups) {
    for (int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; g < groups; g += int64_t(gridDim.x) * blockDim.x) {
        int64_t xi, yi;
        offsets<uint32_t, 1>(px, g * kGroup, &xi);
        offsets<uint32_t, 1>(py, g * kGroup, &yi);
        float v[kGroup];
        ld8<TX>(x, xi, true, v);
        st8<TY>(y, yi, v);
    }
}
template <typename I>
__global__ void copy_planned(LtxView x, LtxView y, Plan px, Plan py, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t xi, yi;
        offsets<I, 1>(px, i, &xi);
        offsets<I, 1>(py, i, &yi);
        if (x.dtype == y.dtype) {
            const int width = (x.dtype == 0 || x.dtype == 5) ? 2 : (x.dtype == 1 || x.dtype == 4) ? 4 : x.dtype == 2 ? 8 : 1;
            if (width == 2) ((uint16_t*)y.data)[yi] = ((const uint16_t*)x.data)[xi];
            else if (width == 4) ((uint32_t*)y.data)[yi] = ((const uint32_t*)x.data)[xi];
            else if (width == 8) ((uint64_t*)y.data)[yi] = ((const uint64_t*)x.data)[xi];
            else ((uint8_t*)y.data)[yi] = ((const uint8_t*)x.data)[xi];
        } else write(y, yi, read(x, xi));
    }
}
__global__ void fill(LtxView y,int64_t n,float start,float step) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n)write(y,address(y,i,y),start+float(i)*step);
}
// fill() of a contiguous y: element i is at offset i.
__global__ void fill_contiguous(LtxView y,int64_t n,float start,float step) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n)write(y,i,start+float(i)*step);
}
// Each op's expression, shared by the generic kernel and the specialized ones below so that every
// instance computes (and nvcc contracts) it the same way. ew_binary covers the ops of x and v only.
__device__ __forceinline__ float ew_unary(int op,float x,float scalar) {
    float z=0;
    if(op==0)z=-x;
    else if(op==1)z=1.0f/(1.0f+expf(-x));
    else if(op==2)z=rsqrtf(x);
    else if(op==3)z=x/(1.0f+expf(-x));
    else if(op==4) {
        constexpr float beta=M_SQRT2*M_2_SQRTPI*float(0.5);
        float cube=x*x*x;
        z=0.5f*x*(1.0f+tanhf(beta*(x+0.044715f*cube)));
    } else if(op==5)z=x*x;
    else if(op==6)z=sqrtf(x);
    else if(op==7)z=expf(x);
    else if(op==8)z=cosf(x);
    else if(op==9)z=sinf(x);
    else if(op==10)z=powf(x,scalar);
    else if(op==11)z=logf(x);
    else if(op==12)z=1.0f/x;
    return z;
}
__device__ __forceinline__ float ew_binary(int op,float x,float v,float scalar,float alpha,int scalar_b) {
    float z=0;
    if(op==20) z=x+alpha*v;
    else if(op==21)z=__fmul_rn(x,v);
    else if(op==22)z=x-alpha*v;
    else if(op==23)z=scalar_b?__fmul_rn(x,1.0f/v):__fdiv_rn(x,v);
    else if(op==24)z=v-alpha*x;
    else if(op==30)z=x>v?1.0f:0.0f;
    else if(op==31)z=x!=0?0.0f:-INFINITY;
    else if(op==32)z=x>=v?1.0f:0.0f;
    else if(op==33)z=x<v?1.0f:0.0f;
    else if(op==35)z=atan2f(x,v);
    else if(op==36)z=isnan(x)?x:fminf(fmaxf(x,scalar),alpha);
    return z;
}
// Operands a, b, c and y at off[0..3] (from the output-indexed plan).
template <typename I>
__global__ void elementwise(int op,LtxView a,LtxView b,LtxView c,LtxView y,Plan p,
                            int64_t n,float scalar,float alpha,int scalar_b) {
  for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; i<n; i+=int64_t(gridDim.x)*blockDim.x) {
    int64_t off[4];
    offsets<I,4>(p,i,off);
    const float x=read(a,off[0]);
    float z=0;
    if(op>=20) {
        float v=scalar_b?scalar:read(b,off[1]);
        if(op==25)z=fmaf(alpha*v,read(c,off[2]),x);
        else if(op==34)z=x!=0?v:read(c,off[2]);
        else if(op==37){
            float weight=alpha!=0?read(c,off[2]):scalar;
            float difference=__fsub_rn(v,x);
            z=fabsf(weight)<0.5f?fmaf(weight,difference,x):fmaf(-__fsub_rn(1.0f,weight),difference,v);
        }
        else z=ew_binary(op,x,v,scalar,alpha,scalar_b);
    } else z=ew_unary(op,x,scalar);
    write(y,off[3],z);
  }
}
// The generic kernel for one op and bf16/fp32 operand types known at compile time (no per-element
// op or dtype dispatch); 32-bit indexing.
template <typename T> __device__ __forceinline__ float ld(const void* p,int64_t i);
template <> __device__ __forceinline__ float ld<__nv_bfloat16>(const void* p,int64_t i){return __bfloat162float(((const __nv_bfloat16*)p)[i]);}
template <> __device__ __forceinline__ float ld<float>(const void* p,int64_t i){return ((const float*)p)[i];}
template <typename T> __device__ __forceinline__ void st(void* p,int64_t i,float x);
template <> __device__ __forceinline__ void st<__nv_bfloat16>(void* p,int64_t i,float x){((__nv_bfloat16*)p)[i]=__float2bfloat16_rn(x);}
template <> __device__ __forceinline__ void st<float>(void* p,int64_t i,float x){((float*)p)[i]=x;}
template <int OP,typename TA,typename TB,typename TY,bool SB>
__global__ void elementwise_typed(const void* a,const void* b,void* y,Plan p,int64_t n,float scalar,float alpha) {
  for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; i<n; i+=int64_t(gridDim.x)*blockDim.x) {
    int64_t off[4];
    offsets<uint32_t,4>(p,i,off);
    const float x=ld<TA>(a,off[0]);
    float z;
    if constexpr (OP>=20) z=ew_binary(OP,x,SB?scalar:ld<TB>(b,off[1]),scalar,alpha,SB);
    else z=ew_unary(OP,x,scalar);
    st<TY>(y,off[3],z);
  }
}
// elementwise_typed eight elements at a time (group_units' bits: 1 a, 2 b, 8 y).
template <int OP,typename TA,typename TB,typename TY,bool SB>
__global__ void elementwise_group(const void* a,const void* b,void* y,Plan p,int64_t groups,unsigned unit,float scalar,float alpha) {
  for (int64_t g=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; g<groups; g+=int64_t(gridDim.x)*blockDim.x) {
    int64_t off[4];
    offsets<uint32_t,4>(p,g*kGroup,off);
    float x[kGroup],v[kGroup],z[kGroup];
    ld8<TA>(a,off[0],unit&1,x);
    if constexpr (OP>=20 && !SB) ld8<TB>(b,off[1],unit&2,v);
    #pragma unroll
    for (int j=0;j<kGroup;++j) {
        if constexpr (OP>=20) z[j]=ew_binary(OP,x[j],SB?scalar:v[j],scalar,alpha,SB);
        else z[j]=ew_unary(OP,x[j],scalar);
    }
    st8<TY>(y,off[3],z);
  }
}
template <int OP,typename TA,typename TB,typename TY,bool SB>
void launch_typed(const LtxView& a,const LtxView& b,const LtxView& y,const Plan& p,int64_t n,float scalar,float alpha,
                  unsigned unit,cudaStream_t s){
    if(unit!=kNoGroups)
        elementwise_group<OP,TA,TB,TY,SB><<<grid_for(n/kGroup),256,0,s>>>(a.data,b.data,y.data,p,n/kGroup,unit,scalar,alpha);
    else elementwise_typed<OP,TA,TB,TY,SB><<<grid_for(n),256,0,s>>>(a.data,b.data,y.data,p,n,scalar,alpha);
}
// dtype 0 bf16, 1 fp32.
template <int OP,typename TA,typename TB,bool SB>
bool typed_y(const LtxView& a,const LtxView& b,const LtxView& y,const Plan& p,int64_t n,float scalar,float alpha,unsigned unit,cudaStream_t s){
    if(y.dtype==0)launch_typed<OP,TA,TB,__nv_bfloat16,SB>(a,b,y,p,n,scalar,alpha,unit,s);
    else if(y.dtype==1)launch_typed<OP,TA,TB,float,SB>(a,b,y,p,n,scalar,alpha,unit,s);
    else return false;
    return true;
}
template <int OP,typename TA,bool SB>
bool typed_b(const LtxView& a,const LtxView& b,const LtxView& y,const Plan& p,int64_t n,float scalar,float alpha,unsigned unit,cudaStream_t s){
    if(SB || OP<20)return typed_y<OP,TA,float,SB>(a,b,y,p,n,scalar,alpha,unit,s);
    if(b.dtype==0)return typed_y<OP,TA,__nv_bfloat16,SB>(a,b,y,p,n,scalar,alpha,unit,s);
    if(b.dtype==1)return typed_y<OP,TA,float,SB>(a,b,y,p,n,scalar,alpha,unit,s);
    return false;
}
template <int OP,bool SB>
bool typed_a(const LtxView& a,const LtxView& b,const LtxView& y,const Plan& p,int64_t n,float scalar,float alpha,unsigned unit,cudaStream_t s){
    if(a.dtype==0)return typed_b<OP,__nv_bfloat16,SB>(a,b,y,p,n,scalar,alpha,unit,s);
    if(a.dtype==1)return typed_b<OP,float,SB>(a,b,y,p,n,scalar,alpha,unit,s);
    return false;
}
// The typed kernel when the op and operand types have one; false otherwise.
bool elementwise_fast(int op,const LtxView& a,const LtxView& b,const LtxView& y,const Plan& p,int64_t n,float scalar,float alpha,
                      int scalar_b,unsigned unit,cudaStream_t s){
    if(n>INT32_MAX)return false;
#define LTX_UNARY(OP) case OP: return typed_a<OP,false>(a,b,y,p,n,scalar,alpha,unit,s);
#define LTX_BINARY(OP) case OP: return scalar_b?typed_a<OP,true>(a,b,y,p,n,scalar,alpha,unit,s):typed_a<OP,false>(a,b,y,p,n,scalar,alpha,unit,s);
    switch(op){
    LTX_UNARY(0) LTX_UNARY(1) LTX_UNARY(2) LTX_UNARY(3) LTX_UNARY(4) LTX_UNARY(5) LTX_UNARY(6) LTX_UNARY(7)
    LTX_UNARY(8) LTX_UNARY(9) LTX_UNARY(10) LTX_UNARY(11) LTX_UNARY(12)
    LTX_BINARY(20) LTX_BINARY(21) LTX_BINARY(22) LTX_BINARY(23)
    default: return false;
    }
#undef LTX_UNARY
#undef LTX_BINARY
}
// The generic kernel's three-operand ops (addcmul 25, where 34, lerp 37 with a tensor weight)
// with typed operands, eight elements at a time; each element is the generic kernel's expression.
template <int OP,typename TA,typename TB,typename TC,typename TY>
__global__ void ternary_group(const void* a,const void* b,const void* c,void* y,Plan p,int64_t groups,unsigned unit,float alpha) {
  for (int64_t g=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; g<groups; g+=int64_t(gridDim.x)*blockDim.x) {
    int64_t off[4];
    offsets<uint32_t,4>(p,g*kGroup,off);
    float x[kGroup],v[kGroup],w[kGroup],z[kGroup];
    ld8<TA>(a,off[0],unit&1,x);
    ld8<TB>(b,off[1],unit&2,v);
    ld8<TC>(c,off[2],unit&4,w);
    #pragma unroll
    for (int j=0;j<kGroup;++j) {
        if constexpr (OP==25) z[j]=fmaf(alpha*v[j],w[j],x[j]);
        else if constexpr (OP==34) z[j]=x[j]!=0?v[j]:w[j];
        else {
            const float weight=w[j];
            const float difference=__fsub_rn(v[j],x[j]);
            z[j]=fabsf(weight)<0.5f?fmaf(weight,difference,x[j]):fmaf(-__fsub_rn(1.0f,weight),difference,v[j]);
        }
    }
    st8<TY>(y,off[3],z);
  }
}
template <int OP,typename TA,typename TB,typename TC>
bool ternary_y(const LtxView& a,const LtxView& b,const LtxView& c,const LtxView& y,const Plan& p,int64_t n,float alpha,unsigned unit,cudaStream_t s){
    if(y.dtype==0)ternary_group<OP,TA,TB,TC,__nv_bfloat16><<<grid_for(n/kGroup),256,0,s>>>(a.data,b.data,c.data,y.data,p,n/kGroup,unit,alpha);
    else if(y.dtype==1)ternary_group<OP,TA,TB,TC,float><<<grid_for(n/kGroup),256,0,s>>>(a.data,b.data,c.data,y.data,p,n/kGroup,unit,alpha);
    else return false;
    return true;
}
template <int OP,typename TA,typename TB>
bool ternary_c(const LtxView& a,const LtxView& b,const LtxView& c,const LtxView& y,const Plan& p,int64_t n,float alpha,unsigned unit,cudaStream_t s){
    if(c.dtype==0)return ternary_y<OP,TA,TB,__nv_bfloat16>(a,b,c,y,p,n,alpha,unit,s);
    if(c.dtype==1)return ternary_y<OP,TA,TB,float>(a,b,c,y,p,n,alpha,unit,s);
    return false;
}
template <int OP,typename TA>
bool ternary_b(const LtxView& a,const LtxView& b,const LtxView& c,const LtxView& y,const Plan& p,int64_t n,float alpha,unsigned unit,cudaStream_t s){
    if(b.dtype==0)return ternary_c<OP,TA,__nv_bfloat16>(a,b,c,y,p,n,alpha,unit,s);
    if(b.dtype==1)return ternary_c<OP,TA,float>(a,b,c,y,p,n,alpha,unit,s);
    return false;
}
template <int OP>
bool ternary_a(const LtxView& a,const LtxView& b,const LtxView& c,const LtxView& y,const Plan& p,int64_t n,float alpha,unsigned unit,cudaStream_t s){
    if constexpr (OP==34) return a.dtype==3 && ternary_b<OP,uint8_t>(a,b,c,y,p,n,alpha,unit,s);
    else {
        if(a.dtype==0)return ternary_b<OP,__nv_bfloat16>(a,b,c,y,p,n,alpha,unit,s);
        if(a.dtype==1)return ternary_b<OP,float>(a,b,c,y,p,n,alpha,unit,s);
        return false;
    }
}
bool ternary_fast(int op,const LtxView& a,const LtxView& b,const LtxView& c,const LtxView& y,const Plan& p,int64_t n,
                  float alpha,int scalar_b,unsigned unit,cudaStream_t s){
    if(n>INT32_MAX || unit==kNoGroups || scalar_b)return false;
    if(op==25)return ternary_a<25>(a,b,c,y,p,n,alpha,unit,s);
    if(op==34)return ternary_a<34>(a,b,c,y,p,n,alpha,unit,s);
    if(op==37 && alpha!=0)return ternary_a<37>(a,b,c,y,p,n,alpha,unit,s);
    return false;
}
// A fused run of eager elementwise ops over one output shape: each element runs the ops in order
// in registers, rounding every op's value to that op's eager output dtype (what storing and
// re-reading its tensor does), with the expressions above. Only the tensors that outlive the run
// are stored. Registers: inputs 0..nin, then one per op.
constexpr int kFuseIn=8,kFuseOut=4,kFuseOps=24,kFuseOperands=kFuseIn+kFuseOut;
using FuseOp=LtxFuseOp;
struct FuseProgram {
    int rank,nin,nout,nops; int64_t dims[8]; int shift[8];
    const void* data[kFuseOperands]; int dtype[kFuseOperands]; int64_t st[kFuseOperands][8];  // nin inputs, then nout outputs
    int out_reg[kFuseOut]; FuseOp ops[kFuseOps]; unsigned unit;
};
__device__ __forceinline__ float fuse_apply(const FuseOp& o,const float* r) {
    const float x=r[o.a];
    if(o.op==50)return x;
    if(o.op<20)return ew_unary(o.op,x,o.scalar);
    if(o.op==25)return fmaf(o.alpha*r[o.b],r[o.c],x);
    return ew_binary(o.op,x,o.scalar_b?o.scalar:r[o.b],o.scalar,o.alpha,o.scalar_b);
}
__device__ __forceinline__ void fuse_load(const void* p,int dtype,int64_t i,bool unit,float* v,int e) {
    if(e==1){v[0]=dtype==0?__bfloat162float(((const __nv_bfloat16*)p)[i]):((const float*)p)[i];return;}
    if(!unit){const float s=dtype==0?__bfloat162float(((const __nv_bfloat16*)p)[i]):((const float*)p)[i];v[0]=v[1]=v[2]=v[3]=s;return;}
    if(dtype==0){
        const uint2 r=*(const uint2*)((const __nv_bfloat16*)p+i);const __nv_bfloat16* h=(const __nv_bfloat16*)&r;
        v[0]=__bfloat162float(h[0]);v[1]=__bfloat162float(h[1]);v[2]=__bfloat162float(h[2]);v[3]=__bfloat162float(h[3]);
    }else{const float4 r=*(const float4*)((const float*)p+i);v[0]=r.x;v[1]=r.y;v[2]=r.z;v[3]=r.w;}
}
__device__ __forceinline__ void fuse_store(void* p,int dtype,int64_t i,const float* v,int e) {
    if(e==1){if(dtype==0)((__nv_bfloat16*)p)[i]=__float2bfloat16_rn(v[0]);else ((float*)p)[i]=v[0];return;}
    if(dtype==0){
        uint2 r;__nv_bfloat16* h=(__nv_bfloat16*)&r;
        h[0]=__float2bfloat16_rn(v[0]);h[1]=__float2bfloat16_rn(v[1]);h[2]=__float2bfloat16_rn(v[2]);h[3]=__float2bfloat16_rn(v[3]);
        *(uint2*)((__nv_bfloat16*)p+i)=r;
    }else *(float4*)((float*)p+i)=make_float4(v[0],v[1],v[2],v[3]);
}
// E consecutive elements per thread: 4 when every operand is unit-stride or broadcast across the
// inner dim (p.unit: unit-stride operands), else 1.
template <int E>
__global__ void fused_elementwise(FuseProgram p,int64_t count) {
  const int operands=p.nin+p.nout;
  for (int64_t g=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; g<count; g+=int64_t(gridDim.x)*blockDim.x) {
    int64_t off[kFuseOperands];
    for(int k=0;k<operands;++k)off[k]=0;
    uint32_t rest=uint32_t(g*E);
    for(int d=p.rank-1;d>=0;--d){
        uint32_t c;
        if(d==0)c=rest;
        else if(p.shift[d]>=0){c=rest&uint32_t(p.dims[d]-1);rest>>=p.shift[d];}
        else{c=rest%uint32_t(p.dims[d]);rest/=uint32_t(p.dims[d]);}
        for(int k=0;k<operands;++k)off[k]+=int64_t(c)*p.st[k][d];
    }
    float in[kFuseIn][E],out[kFuseOut][E];
    for(int k=0;k<p.nin;++k)fuse_load(p.data[k],p.dtype[k],off[k],(p.unit>>k)&1,in[k],E);
    #pragma unroll
    for(int j=0;j<E;++j){
        float r[kFuseIn+kFuseOps];
        for(int k=0;k<p.nin;++k)r[k]=in[k][j];
        for(int q=0;q<p.nops;++q){
            float z=fuse_apply(p.ops[q],r);
            if(p.ops[q].round==0)z=__bfloat162float(__float2bfloat16_rn(z));
            r[p.nin+q]=z;
        }
        for(int k=0;k<p.nout;++k)out[k][j]=r[p.out_reg[k]];
    }
    for(int k=0;k<p.nout;++k)fuse_store(const_cast<void*>(p.data[p.nin+k]),p.dtype[p.nin+k],off[p.nin+k],out[k],E);
  }
}
__global__ void index_tensor(int op,LtxView x,LtxView indices,LtxView y,int dim,int64_t n) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    int64_t linear=i,xi=0;
    for(int d=y.rank-1;d>=0;--d){
        int64_t coordinate=linear%y.dims[d];linear/=y.dims[d];
        int j=d-(y.rank-x.rank);
        if(j<0)continue;
        if(op==0 && d==dim)coordinate=((const int64_t*)indices.data)[address(indices,i,y)];
        else if(op==1)coordinate%=x.dims[j];
        if(coordinate<0 || coordinate>=x.dims[j])asm("trap;");
        xi+=coordinate*x.strides[j];
    }
    int width=(x.dtype==0 || x.dtype==5)?2:(x.dtype==1 || x.dtype==4)?4:x.dtype==2?8:1;
    int64_t yi=address(y,i,y);
    for(int b=0;b<width;b++)((uint8_t*)y.data)[yi*width+b]=((const uint8_t*)x.data)[xi*width+b];
}
__global__ void spatial(LtxView x,LtxView y,Pads pads,int mode,float value,int64_t n) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    int64_t linear=i,xi=0;bool inside=true;
    for(int d=y.rank-1;d>=0;--d){
        int64_t c=linear%y.dims[d];linear/=y.dims[d];
        c=mode==2?c*x.dims[d]/y.dims[d]:c-pads.before[d];
        if(mode==1)c=min(max(c,int64_t(0)),x.dims[d]-1);
        if(c<0 || c>=x.dims[d])inside=false;
        xi+=c*x.strides[d];
    }
    if(inside){
        int width=(x.dtype==0 || x.dtype==5)?2:(x.dtype==1 || x.dtype==4)?4:x.dtype==2?8:1;
        int64_t yi=address(y,i,y);
        for(int b=0;b<width;b++)((uint8_t*)y.data)[yi*width+b]=((const uint8_t*)x.data)[xi*width+b];
    }else write(y,address(y,i,y),value);
}
__global__ void binary_sort(LtxView x,LtxView y,int descending) {
    if(threadIdx.x)return;
    int64_t row=blockIdx.x,n=x.dims[1],out=0;
    for(int pass=0;pass<2;pass++){
        int wanted=descending?1-pass:pass;
        for(int64_t i=0;i<n;i++){
            auto value=((const int32_t*)x.data)[row*x.strides[0]+i*x.strides[1]];
            if(value!=0 && value!=1)asm("trap;");
            if(value==wanted)((int64_t*)y.data)[row*y.strides[0]+out++]=i;
        }
    }
}

template<class T> __device__ float loadf(T x){return float(x);}
template<class T> __device__ T castf(float x){return T(x);}
template<> __device__ __nv_bfloat16 castf(float x){return __float2bfloat16_rn(x);}
template<class T> struct alignas(sizeof(T)*4) Vec {T v[4];};
// Vectorized PyTorch LayerNorm Welford order, also used by the existing H3
// text-encoder port. Mean/variance are accumulated in FP32 for BF16 inputs.
struct Welford {float mean,m2,count;};
__device__ Welford online(float value,Welford a) {
    float delta=value-a.mean,count=a.count+1.0f;
    float mean=a.mean+delta*(1.0f/count);
    return {mean,a.m2+delta*(value-mean),count};
}
__device__ Welford combine(Welford b,Welford a) {
    float delta=b.mean-a.mean,count=a.count+b.count;
    if(count<=0)return {0,0,count};
    float reciprocal=1.0f/count,na=a.count*reciprocal,nb=b.count*reciprocal;
    return {na*a.mean+nb*b.mean,a.m2+b.m2+delta*delta*a.count*nb,count};
}
template<class T> __global__ void layer_norm(const T* x,const T* weight,const T* bias,T* y,
                                             float* mean_out,float* rstd_out,int cols,float eps) {
    __shared__ float buffer[8];
    int lane=threadIdx.x,warp=threadIdx.y,tid=lane+32*warp;
    int64_t offset=int64_t(blockIdx.x)*cols;
    Welford state{0,0,0};
    for(int vector=tid;vector<cols/4;vector+=128) {
#pragma unroll
        for(int j=0;j<4;j++)state=online(loadf(x[offset+vector*4+j]),state);
    }
    for(int delta=16;delta;delta>>=1) {
        Welford other{__shfl_down_sync(0xffffffff,state.mean,delta),
            __shfl_down_sync(0xffffffff,state.m2,delta),__shfl_down_sync(0xffffffff,state.count,delta)};
        state=combine(state,other);
    }
    for(int delta=2;delta;delta>>=1) {
        if(!lane && warp>=delta && warp<2*delta) {
            buffer[2*(warp-delta)]=state.mean;buffer[2*(warp-delta)+1]=state.m2;
            buffer[4+warp-delta]=state.count;
        }
        __syncthreads();
        if(!lane && warp<delta)state=combine(state,{buffer[2*warp],buffer[2*warp+1],buffer[4+warp]});
        __syncthreads();
    }
    if(tid==0){buffer[0]=state.mean;buffer[1]=state.m2/float(cols);}
    __syncthreads();
    float rstd=rsqrtf(buffer[1]+eps);
    if(tid==0){if(mean_out)mean_out[blockIdx.x]=buffer[0];if(rstd_out)rstd_out[blockIdx.x]=rstd;}
    for(int col=tid;col<cols;col+=128) {
        float gamma=weight?loadf(weight[col]):1.0f,beta=bias?loadf(bias[col]):0.0f;
        y[offset+col]=castf<T>(gamma*(rstd*(loadf(x[offset+col])-buffer[0]))+beta);
    }
}
template<class T> __global__ void rms(const T* x,const T* w,T* y,float* rstd,int n,float eps) {
    __shared__ float buffer[6];
    const int tid=threadIdx.x+threadIdx.y*32;
    const Vec<T>* xv=(const Vec<T>*)(x+int64_t(blockIdx.x)*n);
    float sum=0;
    for(int i=tid;i<n/4;i+=128) {
        Vec<T> values=xv[i];
#pragma unroll
        for(int j=0;j<4;j++){float value=loadf(values.v[j]);sum=sum+value*value;}
    }
    for(int offset=16;offset>0;offset>>=1)sum=sum+__shfl_down_sync(0xffffffff,sum,offset);
    for(int offset=2;offset>0;offset/=2) {
        if(threadIdx.x==0 && threadIdx.y>=offset && threadIdx.y<2*offset)
            buffer[2*(threadIdx.y-offset)+1]=sum;
        __syncthreads();
        if(threadIdx.x==0 && threadIdx.y<offset)sum=sum+buffer[2*threadIdx.y+1];
        __syncthreads();
    }
    if(tid==0)buffer[1]=sum/float(n);
    __syncthreads();
    float inv=rsqrtf(buffer[1]+eps);
    if(tid==0 && rstd)rstd[blockIdx.x]=inv;
    Vec<T>* yv=(Vec<T>*)(y+int64_t(blockIdx.x)*n);
    const Vec<T>* wv=(const Vec<T>*)w;
    for(int i=tid;i<n/4;i+=128) {
        Vec<T> values=xv[i],out;
#pragma unroll
        for(int j=0;j<4;j++)out.v[j]=castf<T>(w?loadf(wv[i].v[j])*(inv*loadf(values.v[j])):inv*loadf(values.v[j]));
        yv[i]=out;
    }
}
// rms() for rows of at most 32 vectors: one warp per row. rms() gives such a row's vectors to the
// first warp's lanes (one each) and adds only zeros from its other lanes and warps, so a single
// warp with the same per-lane sum and shuffle steps computes the same value.
template<class T> __global__ void rms_rows(const T* x,const T* w,T* y,float* rstd,int64_t rows,int n,float eps) {
    const int64_t row=int64_t(blockIdx.x)*blockDim.y+threadIdx.y;
    if(row>=rows)return;
    const int lane=threadIdx.x;
    const Vec<T>* xv=(const Vec<T>*)(x+row*n);
    float sum=0;
    Vec<T> values{};
    if(lane<n/4){
        values=xv[lane];
#pragma unroll
        for(int j=0;j<4;j++){float value=loadf(values.v[j]);sum=sum+value*value;}
    }
    for(int offset=16;offset>0;offset>>=1)sum=sum+__shfl_down_sync(0xffffffff,sum,offset);
    sum=__shfl_sync(0xffffffff,sum,0);
    const float inv=rsqrtf(sum/float(n)+eps);
    if(lane==0 && rstd)rstd[row]=inv;
    if(lane<n/4){
        Vec<T>* yv=(Vec<T>*)(y+row*n);
        const Vec<T>* wv=(const Vec<T>*)w;
        Vec<T> out;
#pragma unroll
        for(int j=0;j<4;j++)out.v[j]=castf<T>(w?loadf(wv[lane].v[j])*(inv*loadf(values.v[j])):inv*loadf(values.v[j]));
        yv[lane]=out;
    }
}
}
extern "C" int ltx25_tensor_copy(LtxView x,LtxView y,int64_t n,cudaStream_t stream) {
    if(!n)return cudaGetLastError();
    const LtxView* xs[1]={&x};
    const LtxView* ys[1]={&y};
    const Plan px=make_plan(x,xs,1),py=make_plan(y,ys,1);
    const bool used[1]={true};
    if(n<=INT32_MAX && x.dtype==y.dtype){
        const int width=(x.dtype==0 || x.dtype==5)?2:(x.dtype==1 || x.dtype==4)?4:x.dtype==2?8:1,per=16/width;
        if(group_units(px,xs,used,1,per)==1 && group_units(py,ys,used,1,per)==1){
            copy_chunks<<<grid_for(n/per),256,0,stream>>>((const uint8_t*)x.data,(uint8_t*)y.data,px,py,n/per,per,width);
            return cudaGetLastError();
        }
    }
    if(n<=INT32_MAX && (x.dtype==0 || x.dtype==1) && (y.dtype==0 || y.dtype==1) && x.dtype!=y.dtype
       && group_units(px,xs,used,1,kGroup)==1 && group_units(py,ys,used,1,kGroup)==1){
        if(x.dtype==0)convert_group<__nv_bfloat16,float><<<grid_for(n/kGroup),256,0,stream>>>(x.data,y.data,px,py,n/kGroup);
        else convert_group<float,__nv_bfloat16><<<grid_for(n/kGroup),256,0,stream>>>(x.data,y.data,px,py,n/kGroup);
        return cudaGetLastError();
    }
    if(n<=INT32_MAX)copy_planned<uint32_t><<<grid_for(n),256,0,stream>>>(x,y,px,py,n);
    else copy_planned<int64_t><<<grid_for(n),256,0,stream>>>(x,y,px,py,n);
    return cudaGetLastError();
}
extern "C" int ltx25_tensor_index(int op,LtxView x,LtxView indices,LtxView y,int dim,int64_t n,cudaStream_t stream) {
    if(n)index_tensor<<<(n+255)/256,256,0,stream>>>(op,x,indices,y,dim,n);
    return cudaGetLastError();
}
extern "C" int ltx25_tensor_spatial(LtxView x,LtxView y,const int64_t* before,int mode,float value,int64_t n,cudaStream_t stream) {
    Pads pads{};for(int i=0;i<8;i++)pads.before[i]=before[i];
    if(n)spatial<<<(n+255)/256,256,0,stream>>>(x,y,pads,mode,value,n);return cudaGetLastError();
}
extern "C" int ltx25_tensor_binary_sort(LtxView x,LtxView y,int descending,cudaStream_t stream) {
    binary_sort<<<x.dims[0],1,0,stream>>>(x,y,descending);return cudaGetLastError();
}
extern "C" int ltx25_tensor_fill(LtxView y,int64_t n,float start,float step,cudaStream_t stream) {
    bool contiguous=true;int64_t expected=1;
    for(int d=y.rank-1;d>=0;--d){if(y.dims[d]>1 && y.strides[d]!=expected)contiguous=false;expected*=y.dims[d];}
    if(n && contiguous)fill_contiguous<<<(n+255)/256,256,0,stream>>>(y,n,start,step);
    else if(n)fill<<<(n+255)/256,256,0,stream>>>(y,n,start,step);
    return cudaGetLastError();
}
extern "C" int ltx25_tensor_elementwise(int op,LtxView a,LtxView b,LtxView c,LtxView y,
    int64_t n,float scalar,float alpha,int scalar_b,cudaStream_t stream) {
    if(!n)return cudaGetLastError();
    // Operands the op reads constrain the merge; the others index as zero-stride.
    const bool uses_b=op>=20 && !scalar_b, uses_c=op==25 || op==34 || (op==37 && alpha!=0);
    LtxView none{};none.rank=0;
    const LtxView* views[4]={&a,uses_b?&b:&none,uses_c?&c:&none,&y};
    const Plan p=make_plan(y,views,4);
    const bool used[4]={true,uses_b,uses_c,true};
    unsigned unit=group_units(p,views,used,4,kGroup);
    if(unit!=kNoGroups && !(unit&8))unit=kNoGroups;
    if(elementwise_fast(op,a,b,y,p,n,scalar,alpha,scalar_b,unit,stream))return cudaGetLastError();
    if(ternary_fast(op,a,b,c,y,p,n,alpha,scalar_b,unit,stream))return cudaGetLastError();
    if(n<=INT32_MAX)elementwise<uint32_t><<<grid_for(n),256,0,stream>>>(op,a,b,c,y,p,n,scalar,alpha,scalar_b);
    else elementwise<int64_t><<<grid_for(n),256,0,stream>>>(op,a,b,c,y,p,n,scalar,alpha,scalar_b);
    return cudaGetLastError();
}
// Driver entry points (driver.cpp resolves them at run time).
extern "C" int cuModuleLoadData(void** module, const void* image);
extern "C" int cuCtxGetCurrent(void** context);
extern "C" int cuModuleGetFunction(void** function, void* module, const char* name);
extern "C" int cuLaunchKernel(void* function, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
    unsigned shared, void* stream, void** params, void** extra);
namespace {
// fused_elementwise compiled for one program: the ops, their operand registers, the dtypes and
// the grouped/broadcast form of every input become straight-line code (no register file in local
// memory, no op dispatch). Each op's expression is the one ew_unary/ew_binary/fuse_apply computes,
// with its scalar and alpha still read from the program (same operands, same contraction).
const char* kFusedPrelude = R"(
#define kFuseIn 8
#define kFuseOut 4
#define kFuseOps 24
#define kFuseOperands 12
struct FuseOp {int op,a,b,c,round;float scalar,alpha;int scalar_b;};
struct FuseProgram {
    int rank,nin,nout,nops; long long dims[8]; int shift[8];
    const void* data[kFuseOperands]; int dtype[kFuseOperands]; long long st[kFuseOperands][8];
    int out_reg[kFuseOut]; FuseOp ops[kFuseOps]; unsigned unit;
};
__device__ __forceinline__ float bf(unsigned short h){return __uint_as_float(((unsigned)h)<<16);}
__device__ __forceinline__ unsigned short tobf(float x){unsigned short r;asm("cvt.rn.bf16.f32 %0, %1;":"=h"(r):"f"(x));return r;}
__device__ __forceinline__ float rbf(float x){return bf(tobf(x));}
)";
std::string fused_source(const FuseProgram& p, int e) {
    std::string s = kFusedPrelude;
    auto num = [](long long v) { return std::to_string(v); };
    const int operands = p.nin + p.nout;
    s += "extern \"C\" __global__ void __launch_bounds__(256) fused(const FuseProgram p, long long count) {\n";
    s += " for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x; g < count; g += (long long)gridDim.x * blockDim.x) {\n";
    for (int k = 0; k < operands; ++k) s += "  long long o" + num(k) + " = 0;\n";
    s += "  unsigned rest = (unsigned)(g * " + num(e) + ");\n";
    s += "  for (int d = p.rank - 1; d >= 0; --d) {\n   unsigned c;\n";
    s += "   if (d == 0) c = rest; else if (p.shift[d] >= 0) { c = rest & (unsigned)(p.dims[d] - 1); rest >>= p.shift[d]; }\n";
    s += "   else { c = rest % (unsigned)p.dims[d]; rest /= (unsigned)p.dims[d]; }\n";
    for (int k = 0; k < operands; ++k) s += "   o" + num(k) + " += (long long)c * p.st[" + num(k) + "][d];\n";
    s += "  }\n";
    // Loads: in<k>_<j>.
    for (int k = 0; k < p.nin; ++k) {
        const bool b16 = p.dtype[k] == 0, unit = e == 4 && ((p.unit >> k) & 1);
        const std::string ptr = std::string("((const ") + (b16 ? "unsigned short" : "float") + "*)p.data[" + num(k) + "])";
        if (unit && b16) {
            s += "  uint2 u" + num(k) + " = *(const uint2*)(" + ptr + " + o" + num(k) + ");\n";
            s += "  const unsigned short* h" + num(k) + " = (const unsigned short*)&u" + num(k) + ";\n";
            for (int j = 0; j < e; ++j) s += "  const float in" + num(k) + "_" + num(j) + " = bf(h" + num(k) + "[" + num(j) + "]);\n";
        } else if (unit) {
            s += "  float4 u" + num(k) + " = *(const float4*)(" + ptr + " + o" + num(k) + ");\n";
            const char* lane[4] = {"x", "y", "z", "w"};
            for (int j = 0; j < e; ++j) s += "  const float in" + num(k) + "_" + num(j) + " = u" + num(k) + "." + lane[j] + ";\n";
        } else {
            s += "  const float in" + num(k) + "_0 = " + (b16 ? "bf(" + ptr + "[o" + num(k) + "])" : ptr + "[o" + num(k) + "]") + ";\n";
            for (int j = 1; j < e; ++j) s += "  const float in" + num(k) + "_" + num(j) + " = in" + num(k) + "_0;\n";
        }
    }
    // Ops per element: r<j>_<reg>.
    for (int j = 0; j < e; ++j) {
        const std::string J = num(j);
        auto reg = [&](int r) { return r < p.nin ? "in" + num(r) + "_" + J : "r" + J + "_" + num(r); };
        for (int q = 0; q < p.nops; ++q) {
            const FuseOp& o = p.ops[q];
            const std::string x = reg(o.a), Q = "p.ops[" + num(q) + "]";
            const std::string v = o.op >= 20 && o.op != 50 ? (o.scalar_b ? Q + ".scalar" : reg(o.b)) : "";
            std::string z;
            switch (o.op) {
            case 0: z = "-" + x; break;
            case 1: z = "1.0f/(1.0f+expf(-" + x + "))"; break;
            case 2: z = "rsqrtf(" + x + ")"; break;
            case 3: z = x + "/(1.0f+expf(-" + x + "))"; break;
            case 4: z = "0.5f*" + x + "*(1.0f+tanhf(beta*(" + x + "+0.044715f*(" + x + "*" + x + "*" + x + "))))"; break;
            case 5: z = x + "*" + x; break;
            case 6: z = "sqrtf(" + x + ")"; break;
            case 7: z = "expf(" + x + ")"; break;
            case 8: z = "cosf(" + x + ")"; break;
            case 9: z = "sinf(" + x + ")"; break;
            case 10: z = "powf(" + x + "," + Q + ".scalar)"; break;
            case 11: z = "logf(" + x + ")"; break;
            case 12: z = "1.0f/" + x; break;
            case 20: z = x + "+" + Q + ".alpha*" + v; break;
            case 21: z = "__fmul_rn(" + x + "," + v + ")"; break;
            case 22: z = x + "-" + Q + ".alpha*" + v; break;
            case 23: z = o.scalar_b ? "__fmul_rn(" + x + ",1.0f/" + v + ")" : "__fdiv_rn(" + x + "," + v + ")"; break;
            case 24: z = v + "-" + Q + ".alpha*" + x; break;
            case 25: z = "fmaf(" + Q + ".alpha*" + reg(o.b) + "," + reg(o.c) + "," + x + ")"; break;
            case 50: z = x; break;
            default: return "";
            }
            const std::string name = "r" + J + "_" + num(p.nin + q);
            if (o.op == 4) s += "  float " + name + ";\n  { const float beta = 1.41421356237309504880 * 1.12837916709551257390 * float(0.5);\n    const float cube_x = " + x + ";\n    const float cube = cube_x*cube_x*cube_x;\n    " + name + " = 0.5f*cube_x*(1.0f+tanhf(beta*(cube_x+0.044715f*cube))); }\n";
            else s += "  float " + name + " = " + z + ";\n";
            if (o.round == 0) s += "  " + name + " = rbf(" + name + ");\n";
        }
    }
    // Stores.
    for (int k = 0; k < p.nout; ++k) {
        const int t = p.nin + k;
        const bool b16 = p.dtype[t] == 0;
        auto val = [&](int j) { const int r = p.out_reg[k]; return r < p.nin ? "in" + num(r) + "_" + num(j) : "r" + num(j) + "_" + num(r); };
        const std::string ptr = std::string("((") + (b16 ? "unsigned short" : "float") + "*)p.data[" + num(t) + "])";
        if (e == 4 && b16) {
            s += "  { uint2 w; unsigned short* h = (unsigned short*)&w;";
            for (int j = 0; j < 4; ++j) s += " h[" + num(j) + "] = tobf(" + val(j) + ");";
            s += " *(uint2*)(" + ptr + " + o" + num(t) + ") = w; }\n";
        } else if (e == 4) {
            s += "  { float4 w; w.x = " + val(0) + "; w.y = " + val(1) + "; w.z = " + val(2) + "; w.w = " + val(3) + ";";
            s += " *(float4*)(" + ptr + " + o" + num(t) + ") = w; }\n";
        } else {
            s += "  " + ptr + "[o" + num(t) + "] = " + (b16 ? "tobf(" + val(0) + ")" : val(0)) + ";\n";
        }
    }
    s += " }\n}\n";
    return s;
}
std::mutex& fused_mutex() { static std::mutex m; return m; }
// Loaded kernels by (CUDA context, source): a module and its functions belong to one context.
std::map<std::pair<void*, std::string>, void*>& fused_kernels() {
    static auto* k = new std::map<std::pair<void*, std::string>, void*>(); return *k;
}
// NVRTC's CUBIN for `src`, or empty when it does not compile.
std::string compile_fused(const std::string& src) {
    std::string cubin;
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src.c_str(), "fused.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS) return cubin;
    const char* opts[] = {"--gpu-architecture=sm_100a", "--std=c++17", "-default-device"};
    size_t bytes = 0;
    if (nvrtcCompileProgram(prog, 3, opts) == NVRTC_SUCCESS && nvrtcGetCUBINSize(prog, &bytes) == NVRTC_SUCCESS) {
        cubin.resize(bytes);
        if (nvrtcGetCUBIN(prog, cubin.data()) != NVRTC_SUCCESS) cubin.clear();
    } else if (std::getenv("LTX25_FUSE_LOG")) {
        size_t n = 0; nvrtcGetProgramLogSize(prog, &n); std::string log(n, '\0'); nvrtcGetProgramLog(prog, log.data());
        std::fprintf(stderr, "fused kernel compile failed:\n%s\n%s\n", log.c_str(), src.c_str());
    }
    nvrtcDestroyProgram(&prog);
    return cubin;
}
// LTX25_KERNEL_CACHE=<dir>: compiled programs are kept there as <key>.cubin, the key covering the
// source and the NVRTC version, so later processes (other containers sharing the directory) load
// them instead of compiling. A cubin is written to a temporary name and renamed into place.
std::string cached_fused(const std::string& src) {
    const char* dir = std::getenv("LTX25_KERNEL_CACHE");
    if (!dir || !*dir) return compile_fused(src);
    int major = 0, minor = 0;
    nvrtcVersion(&major, &minor);
    std::ostringstream key;
    key << std::hex << std::hash<std::string>{}(src + "\nnvrtc " + std::to_string(major) + "." + std::to_string(minor) + " sm_100a");
    const std::string path = std::string(dir) + "/fused-" + key.str() + "-" + std::to_string(src.size()) + ".cubin";
    const std::string source_path = path + ".cu";
    {
        std::ifstream cached(path, std::ios::binary), cached_source(source_path, std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(cached)), {}), text((std::istreambuf_iterator<char>(cached_source)), {});
        if (!bytes.empty() && text == src) return bytes;  // the stored source must match exactly (no hash collision)
    }
    std::string cubin = compile_fused(src);
    if (!cubin.empty()) {
        const std::string tmp = path + "." + std::to_string(getpid()) + ".tmp", tmp_source = source_path + "." + std::to_string(getpid()) + ".tmp";
        std::ofstream(tmp_source, std::ios::binary) << src;
        std::ofstream(tmp, std::ios::binary).write(cubin.data(), std::streamsize(cubin.size()));
        std::rename(tmp_source.c_str(), source_path.c_str());
        std::rename(tmp.c_str(), path.c_str());
    }
    return cubin;
}
// The compiled kernel for a program (nullptr when NVRTC or the module load fails: the interpreted
// kernel runs instead).
void* fused_kernel(const FuseProgram& p, int e) {
    const std::string src = fused_source(p, e);
    if (src.empty()) return nullptr;
    void* context = nullptr;
    if (cuCtxGetCurrent(&context) != 0 || !context) return nullptr;
    std::lock_guard<std::mutex> lock(fused_mutex());
    auto& cache = fused_kernels();
    const auto key = std::make_pair(context, src);
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    void* function = nullptr;
    static auto* cubins = new std::map<std::string, std::string>();  // compiled once per process for every context
    auto found = cubins->find(src);
    if (found == cubins->end()) found = cubins->emplace(src, cached_fused(src)).first;
    const std::string& cubin = found->second;
    void* module = nullptr;
    if (!cubin.empty() && cuModuleLoadData(&module, cubin.data()) == 0)
        if (cuModuleGetFunction(&function, module, "fused") != 0) function = nullptr;
    cache[key] = function;
    return function;
}
}  // namespace
// `outputs` share one shape, the iteration space; inputs broadcast to it as in address().
extern "C" int ltx25_fused_elementwise(const LtxView* inputs,int nin,const LtxView* outputs,const int* out_regs,int nout,
    const LtxFuseOp* ops,int nops,int64_t n,cudaStream_t stream) {
    if(nin<1 || nin>kFuseIn || nout<1 || nout>kFuseOut || nops<1 || nops>kFuseOps || n>INT32_MAX)return -1;
    if(!n)return cudaGetLastError();
    for(int k=0;k<nops;++k){
        const LtxFuseOp& o=ops[k];const int defined=nin+k;
        if(o.a<0 || o.a>=defined || o.round<0 || o.round>1)return -1;
        if(o.op>=20 && o.op!=50 && !o.scalar_b && (o.b<0 || o.b>=defined))return -1;
        if(o.op==25 && (o.c<0 || o.c>=defined))return -1;
    }
    const LtxView& by=outputs[0];
    FuseProgram p{};
    p.nin=nin;p.nout=nout;p.nops=nops;
    const int operands=nin+nout;
    int64_t dims[8],st[kFuseOperands][8];int r=0;
    for(int d=0;d<by.rank;++d){
        if(by.dims[d]==1)continue;
        dims[r]=by.dims[d];
        for(int k=0;k<operands;++k){
            const LtxView& v=k<nin?inputs[k]:outputs[k-nin];
            if(v.dtype!=0 && v.dtype!=1)return -1;
            const int j=d-(by.rank-v.rank);
            st[k][r]=j>=0 && v.dims[j]!=1?v.strides[j]:0;
        }
        ++r;
    }
    for(int k=nin;k<operands;++k)
        for(int d=0;d<by.rank;++d)if(outputs[k-nin].rank!=by.rank || outputs[k-nin].dims[d]!=by.dims[d])return -1;
    for(int d=0;d<r;++d){
        bool merge=p.rank>0;
        for(int k=0;merge && k<operands;++k)merge=p.st[k][p.rank-1]==st[k][d]*dims[d];
        if(merge){p.dims[p.rank-1]*=dims[d];for(int k=0;k<operands;++k)p.st[k][p.rank-1]=st[k][d];}
        else{p.dims[p.rank]=dims[d];for(int k=0;k<operands;++k)p.st[k][p.rank]=st[k][d];++p.rank;}
    }
    for(int d=0;d<p.rank;++d){
        p.shift[d]=-1;
        if((p.dims[d]&(p.dims[d]-1))==0)for(int b=0;b<63;++b)if((int64_t(1)<<b)==p.dims[d])p.shift[d]=b;
    }
    for(int k=0;k<operands;++k){
        const LtxView& v=k<nin?inputs[k]:outputs[k-nin];
        p.data[k]=v.data;p.dtype[k]=v.dtype;
    }
    for(int k=0;k<nout;++k)p.out_reg[k]=out_regs[k];
    for(int k=0;k<nops;++k)p.ops[k]=ops[k];
    // Groups of four: inner dim a multiple of 4; each input unit-stride (aligned, outer strides of whole
    // groups) or broadcast across it; each output unit-stride.
    bool grouped=p.rank>0 && p.dims[p.rank-1]%4==0;
    for(int k=0;grouped && k<operands;++k){
        const int64_t inner=p.st[k][p.rank-1];
        if(inner==0 && k<nin)continue;
        if(inner!=1 || reinterpret_cast<uintptr_t>(p.data[k])%(p.dtype[k]==0?8:16)){grouped=false;break;}
        for(int d=0;d<p.rank-1;++d)if(p.st[k][d]%4){grouped=false;break;}
        if(k<nin)p.unit|=1u<<k;
    }
    if(!grouped)p.unit=0;
    const int e=grouped?4:1;
    static const bool compiled=!std::getenv("LTX25_FUSE_INTERPRETED");
    if(void* kernel=compiled?fused_kernel(p,e):nullptr){
        long long count=n/e;void* params[]={&p,&count};
        if(cuLaunchKernel(kernel,grid_for(count),1,1,256,1,1,0,stream,params,nullptr)!=0)return -1;
        return cudaGetLastError();
    }
    if(grouped)fused_elementwise<4><<<grid_for(n/4),256,0,stream>>>(p,n/4);
    else fused_elementwise<1><<<grid_for(n),256,0,stream>>>(p,n);
    return cudaGetLastError();
}
extern "C" int ltx25_tensor_rms(const void* x,const void* w,void* y,float* rstd,
    int64_t rows,int cols,float eps,int dtype,cudaStream_t stream) {
    if(cols%4 || (dtype!=0 && dtype!=1))return -1;
    if(rows && cols<=128) {
        const unsigned blocks=unsigned((rows+7)/8);
        if(dtype==0)rms_rows<<<blocks,dim3(32,8),0,stream>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)w,
            (__nv_bfloat16*)y,rstd,rows,cols,eps);
        else rms_rows<<<blocks,dim3(32,8),0,stream>>>((const float*)x,(const float*)w,(float*)y,rstd,rows,cols,eps);
    }else if(rows) {
        if(dtype==0)rms<<<rows,dim3(32,4),0,stream>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)w,
            (__nv_bfloat16*)y,rstd,cols,eps);
        else rms<<<rows,dim3(32,4),0,stream>>>((const float*)x,(const float*)w,(float*)y,rstd,cols,eps);
    }
    return cudaGetLastError();
}
extern "C" int ltx25_tensor_layer_norm(const void* x,const void* w,const void* bias,void* y,
    float* mean,float* rstd,int64_t rows,int cols,float eps,int dtype,cudaStream_t stream) {
    if(cols%4 || (dtype!=0 && dtype!=1))return -1;
    if(rows) {
        if(dtype==0)layer_norm<<<rows,dim3(32,4),0,stream>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)w,
            (const __nv_bfloat16*)bias,(__nv_bfloat16*)y,mean,rstd,cols,eps);
        else layer_norm<<<rows,dim3(32,4),0,stream>>>((const float*)x,(const float*)w,(const float*)bias,
            (float*)y,mean,rstd,cols,eps);
    }
    return cudaGetLastError();
}
