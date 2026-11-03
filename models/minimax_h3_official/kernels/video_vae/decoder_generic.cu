// Geometry-general native forms of the pinned MiniMax H3 compiled VAE decoder.
// Keep the original FMA, float32 accumulation and final half store boundaries.
// Candidate until the strict operator/full-decoder report records acceptance.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace {
using Half = __half;
struct Args { uint64_t p[10]; };
template<class T> __device__ const T* in(const Args& a,int j){return reinterpret_cast<const T*>(a.p[j]);}
template<class T> __device__ T* out(const Args& a,int j){return reinterpret_cast<T*>(a.p[j]);}
__device__ float hf(Half x){return __half2float(x);}
__device__ Half fh(float x){return __float2half_rn(x);}
__device__ float initial(const Args&a,int row,int col,int patches){
    if(row<patches)return __fadd_rn(hf(in<Half>(a,0)[row*2048+col]),in<float>(a,1)[col]);
    if(row<patches+4)return in<float>(a,2)[(row-patches)*2048+col];
    return 0.f;
}
__device__ float sine(float x){return sinf(x);}
__device__ float cosine(float x){return cosf(x);}

template<int Kind> __device__ float residual(const Args&a,int row,int col,int patches){
    int i=row*2048+col;
    if constexpr(Kind==1)return initial(a,row,col,patches);
    if constexpr(Kind==8)return fmaf(hf(in<Half>(a,3)[i]),in<float>(a,4)[col],initial(a,row,col,patches));
    if constexpr(Kind==13)return fmaf(hf(in<Half>(a,1)[i]),in<float>(a,2)[col],in<float>(a,0)[i]);
    if constexpr(Kind==14){
        float first=fmaf(hf(in<Half>(a,1)[i]),in<float>(a,2)[col],in<float>(a,0)[i]);
        return fmaf(hf(in<Half>(a,3)[i]),in<float>(a,4)[col],first);
    }
    return 0.f;
}

template<int Kind,int R> __global__ void rms(Args a,int patches){
    __shared__ float values[2048];
    __shared__ float partial[R];
    const int row=blockIdx.x;
    for(int r=threadIdx.x;r<R;r+=blockDim.x){
        float sum=0.f;
        #pragma unroll
        for(int off=0;off<2048;off+=R){
            float x=residual<Kind>(a,row,r+off,patches);
            values[r+off]=x;
            sum=fmaf(x,x,sum);
        }
        partial[r]=sum;
    }
    __syncthreads();
    // Original blocked layout reduces four adjacent registers in this order,
    // then warp XOR 16..1, then the four/eight warp partials.
    __shared__ float warp_sums[8];
    float total=0.f;int lane=threadIdx.x%32,warp=threadIdx.x/32;
    if(threadIdx.x<R/4){int r=threadIdx.x*4;
        total=__fadd_rn(__fadd_rn(__fadd_rn(partial[r],partial[r+1]),partial[r+2]),partial[r+3]);}
    for(int off=16;off;off>>=1)total=__fadd_rn(total,__shfl_xor_sync(0xffffffff,total,off));
    if(lane==0)warp_sums[warp]=total;__syncthreads();
    total=lane<R/128?warp_sums[lane]:0.f;
    for(int off=R/256;off;off>>=1)total=__fadd_rn(total,__shfl_xor_sync(0xffffffff,total,off));
    float inv=rsqrtf(__fadd_rn(__fmul_rn(__shfl_sync(0xffffffff,total,0),1.f/2048.f),1.e-5f));
    for(int col=threadIdx.x;col<2048;col+=blockDim.x){
        int i=row*2048+col;
        constexpr int weight=Kind==1?3:Kind==8?5:Kind==13?3:5;
        Half y=fh(__fmul_rn(__fmul_rn(values[col],inv),in<float>(a,weight)[col]));
        if constexpr(Kind==1||Kind==13){out<Half>(a,4)[i]=y;out<Half>(a,5)[i]=y;out<Half>(a,6)[i]=y;}
        if constexpr(Kind==8){out<float>(a,6)[i]=values[col];out<Half>(a,7)[i]=y;}
        if constexpr(Kind==14){out<float>(a,0)[i]=values[col];out<Half>(a,6)[i]=y;}
    }
}

__global__ void qk_sum(Args a,int rows){
    int row=blockIdx.x*16+threadIdx.x/8,lane=threadIdx.x%8;
    float sum=0.f;
    if(row<rows){
        int base=row*64+lane*8;float second=hf(in<Half>(a,0)[base+1]);
        sum=__fmul_rn(second,second);
        for(int j=0;j<8;++j){if(j==1)continue;float x=hf(in<Half>(a,0)[base+j]);sum=fmaf(x,x,sum);}
    }
    for(int off=4;off;off>>=1)sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,off));
    if(!lane&&row<rows)out<float>(a,1)[row]=sum;
}

__global__ void positions(Args a,int t,int h,int w){
    int i=blockIdx.x*blockDim.x+threadIdx.x,patches=t*h*w;
    if(i>=(patches+5)*3)return;
    int token=i/3,axis=i%3;
    float value=0.f;
    if(token<patches){
        int coord=axis==0?token/(h*w):axis==1?token/w%h:token%w;
        int size=axis==0?t:axis==1?h:w;
        float fraction=__fmul_rn(__fadd_rn(float(coord),.5f),__fdiv_rn(1.f,float(size)));
        value=fmaf(fraction,2.f,-1.f);
    }
    out<float>(a,0)[i]=__fmul_rn(value,6.283185307179586f);
}

__global__ void rope_sin(Args a,int count){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    int col=i%48,headrow=i/48,token=headrow/32;
    int partner=col<24?col+24:col-24;
    float angle=__fmul_rn(in<float>(a,2)[token*3+(col/8)%3],in<float>(a,3)[col%8]);
    float s=sine(angle);
    float q=__fmul_rn(hf(in<Half>(a,0)[headrow*64+partner]),rsqrtf(__fadd_rn(__fmul_rn(in<float>(a,1)[headrow],1.f/64.f),1.e-5f)));
    float k=__fmul_rn(hf(in<Half>(a,4)[headrow*64+partner]),rsqrtf(__fadd_rn(__fmul_rn(in<float>(a,5)[headrow],1.f/64.f),1.e-5f)));
    if(col<24){q=-q;k=-k;}
    out<Half>(a,6)[i]=fh(__fmul_rn(q,s));out<Half>(a,7)[i]=fh(__fmul_rn(k,s));
}

__global__ void rope_cos(Args a,int count){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    int col=i%64,headrow=i/64,token=headrow/32;
    float q=__fmul_rn(hf(in<Half>(a,0)[i]),rsqrtf(__fadd_rn(__fmul_rn(in<float>(a,1)[headrow],1.f/64.f),1.e-5f)));
    float k=__fmul_rn(hf(in<Half>(a,5)[i]),rsqrtf(__fadd_rn(__fmul_rn(in<float>(a,6)[headrow],1.f/64.f),1.e-5f)));
    if(col<48){
        float angle=__fmul_rn(in<float>(a,2)[token*3+(col/8)%3],in<float>(a,3)[col%8]);
        float c=cosine(angle);
        q=fmaf(q,c,hf(in<Half>(a,4)[headrow*48+col]));
        k=fmaf(k,c,hf(in<Half>(a,7)[headrow*48+col]));
    }
    out<Half>(a,8)[i]=fh(q);out<Half>(a,9)[i]=fh(k);
}

__device__ float exp_plus_one(float x){
    float e,scale,answer;
    asm volatile("{ .reg .f32 a,b,c,d,r; .reg .b32 bits;\n"
        "fma.rn.ftz.f32 a,%2,0f3BBB989D,0f3F000000;\n"
        "cvt.ftz.sat.f32.f32 b,a;\n"
        "fma.rm.ftz.f32 c,b,0f437C0000,0f4B400001;\n"
        "add.f32 d,c,0fCB40007F;\n"
        "neg.f32 d,d;\n"
        "fma.rn.ftz.f32 r,%2,0f3FB8AA3B,d;\n"
        "fma.rn.ftz.f32 r,%2,0f32A57060,r;\n"
        "mov.b32 bits,c; shl.b32 bits,bits,23; mov.b32 %1,bits;\n"
        "ex2.approx.ftz.f32 %0,r; }":"=f"(e),"=f"(scale):"f"(x));
    answer=fmaf(e,scale,1.f);return answer;
}
__device__ float div_full(float a,float b){float v;asm("div.full.f32 %0,%1,%2;":"=f"(v):"f"(a),"f"(b));return v;}

__global__ void swiglu(Args a,int count){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    int row=i/8192,col=i%8192;
    float x=hf(in<Half>(a,0)[row*16384+col]),g=hf(in<Half>(a,0)[row*16384+8192+col]);
    float gate=div_full(g,exp_plus_one(-g));
    out<Half>(a,1)[i]=fh(__fmul_rn(x,gate));
}

struct Welford{float mean,m2,n;};
__device__ Welford combine(Welford a,Welford b){
    float delta=__fsub_rn(b.mean,a.mean),n=__fadd_rn(a.n,b.n);
    float ratio=div_full(b.n,n);
    float mean=fmaf(delta,ratio,a.mean);
    float m2=fmaf(__fmul_rn(__fmul_rn(delta,delta),a.n),ratio,__fadd_rn(a.m2,b.m2));
    return {mean,m2,n};
}
__global__ void layer_norm(Args a,int rows){
    __shared__ float values[2048];
    __shared__ Welford partial[1024];
    int row=blockIdx.x;
    for(int r=threadIdx.x;r<1024;r+=blockDim.x){
        float x=fmaf(hf(in<Half>(a,0)[row*2048+r]),in<float>(a,2)[r],in<float>(a,1)[row*2048+r]);
        float y=fmaf(hf(in<Half>(a,0)[row*2048+r+1024]),in<float>(a,2)[r+1024],in<float>(a,1)[row*2048+r+1024]);
        values[r]=x;values[r+1024]=y;
        float delta=__fsub_rn(y,x),mean=__fadd_rn(__fmul_rn(delta,.5f),x);
        partial[r]={mean,__fmul_rn(delta,__fsub_rn(y,mean)),2.f};
    }
    __syncthreads();
    __shared__ Welford warp_sums[8];
    int lane=threadIdx.x%32,warp=threadIdx.x/32,r=threadIdx.x*4;
    Welford total=combine(combine(combine(partial[r],partial[r+1]),partial[r+2]),partial[r+3]);
    for(int off=16;off;off>>=1){
        Welford other={__shfl_xor_sync(0xffffffff,total.mean,off),__shfl_xor_sync(0xffffffff,total.m2,off),__shfl_xor_sync(0xffffffff,total.n,off)};
        total=combine(total,other);
    }
    if(!lane)warp_sums[warp]=total;__syncthreads();
    total=warp_sums[lane%8];
    for(int off=4;off;off>>=1){
        Welford other={__shfl_xor_sync(0xffffffff,total.mean,off),__shfl_xor_sync(0xffffffff,total.m2,off),__shfl_xor_sync(0xffffffff,total.n,off)};
        total=combine(total,other);
    }
    float mean=__shfl_sync(0xffffffff,total.mean,0),inv=rsqrtf(__fadd_rn(__fmul_rn(__shfl_sync(0xffffffff,total.m2,0),1.f/2048.f),1.e-5f));
    for(int col=threadIdx.x;col<2048;col+=blockDim.x){
        float normalized=__fmul_rn(__fsub_rn(values[col],mean),inv);
        out<Half>(a,0)[row*2048+col]=fh(fmaf(normalized,in<float>(a,3)[col],in<float>(a,4)[col]));
    }
}

__global__ void unpack(Args a,int t,int h,int w){
    int i=blockIdx.x*blockDim.x+threadIdx.x,count=3*t*4*h*16*w*16;if(i>=count)return;
    int x=i%(w*16),y=i/(w*16)%(h*16),frame=i/(w*h*256)%(t*4),c=i/(w*h*256*t*4);
    int token=((frame/4)*h+y/16)*w+x/16;
    int feature=c*1024+(frame%4)*256+(y%16)*16+x%16;
    out<Half>(a,1)[i]=in<Half>(a,0)[token*3072+feature];
}
int grid(int count){return (count+255)/256;}
}

extern "C" int h3v_generic_decoder_kernel(int kind,const uint64_t* params,int param_count,int T,int H,int W,void* stream){
    if(!params||T!=7||H<2||H>16||W<2||W>16)return cudaErrorInvalidValue;
    int pointers=0,expected=0;
    switch(kind){
        case 1:case 13:case 14:pointers=7;expected=11;break;
        case 4:pointers=2;expected=6;break;
        case 5:pointers=1;expected=4;break;
        case 6:pointers=8;expected=11;break;
        case 7:pointers=10;expected=13;break;
        case 8:pointers=8;expected=12;break;
        case 11:case 18:pointers=2;expected=5;break;
        case 15:pointers=5;expected=9;break;
        default:return cudaErrorInvalidValue;
    }
    if(param_count!=expected)return cudaErrorInvalidValue;
    Args a{};for(int i=0;i<pointers;++i){if(!params[i])return cudaErrorInvalidValue;a.p[i]=params[i];}
    int patches=T*H*W,rows=patches+5;auto s=static_cast<cudaStream_t>(stream);
    switch(kind){
        case 1:rms<1,512><<<rows,256,0,s>>>(a,patches);break;
        case 4:qk_sum<<<(rows*32+15)/16,128,0,s>>>(a,rows*32);break;
        case 5:positions<<<grid(rows*3),256,0,s>>>(a,T,H,W);break;
        case 6:rope_sin<<<grid(rows*32*48),256,0,s>>>(a,rows*32*48);break;
        case 7:rope_cos<<<grid(rows*2048),256,0,s>>>(a,rows*2048);break;
        case 8:rms<8,512><<<rows,256,0,s>>>(a,patches);break;
        case 11:swiglu<<<grid(rows*8192),256,0,s>>>(a,rows*8192);break;
        case 13:rms<13,1024><<<rows,256,0,s>>>(a,patches);break;
        case 14:rms<14,1024><<<rows,256,0,s>>>(a,patches);break;
        case 15:layer_norm<<<rows,256,0,s>>>(a,rows);break;
        case 18:unpack<<<grid(3*T*4*H*16*W*16),256,0,s>>>(a,T,H,W);break;
    }
    return cudaGetLastError();
}
