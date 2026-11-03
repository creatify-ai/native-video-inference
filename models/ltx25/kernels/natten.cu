// Exact NATTEN 0.21.7 cutlass-fna forward headers, with its --use_fast_math build.
// The original NATTEN and CUTLASS licenses remain in the pinned SDK checkout.
#include <tuple>
#include <cuda_runtime.h>
#include <natten/cuda/fna/kernel_forward.h>
namespace fna = natten::cuda::fna;

template<class K>
__global__ void __launch_bounds__(K::kNumThreads,K::kMinBlocksPerSm) kernel(typename K::Params p) {
    if(!p.advance_to_block())return;K::attention_kernel(p);
}
template<int Q,int KV>
int forward(const void* q,const void* k,const void* v,void* out,void* accum,void* lse,
    int batch,int t,int h,int w,int heads,const int* window,const int* qt,const int* kt,float scale,cudaStream_t stream) {
    using K=fna::FusedNeighborhoodAttentionKernel<3,fna::CausalMask<false,false,false>,
        cutlass::bfloat16_t,cutlass::arch::Sm80,true,Q,KV,64>;
    typename K::Params p;
    p.query_ptr=(typename K::scalar_t*)q;p.key_ptr=(typename K::scalar_t*)k;p.value_ptr=(typename K::scalar_t*)v;
    p.output_ptr=(typename K::output_t*)out;
    p.output_accum_ptr=K::kNeedsOutputAccumulatorBuffer?(typename K::output_accum_t*)accum:nullptr;
    p.logsumexp_ptr=(typename K::lse_scalar_t*)lse;
    p.num_heads=heads;p.head_dim=64;p.head_dim_value=64;p.num_batches=batch;
    p.num_queries=typename K::Dim(t,h,w);p.scale=scale;
    p.kernel_size=typename K::Dim(window[0],window[1],window[2]);
    p.stride=typename K::Dim(1,1,1);p.dilation=typename K::Dim(1,1,1);
    p.query_tile_shape=typename K::Dim(qt[0],qt[1],qt[2]);
    p.key_tile_shape=typename K::Dim(kt[0],kt[1],kt[2]);
    auto fn=kernel<K>;constexpr int smem=sizeof(typename K::SharedStorage);
    if(smem>0xc000){auto err=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);if(err)return err;}
    fn<<<p.getBlocksGrid(),p.getThreadsGrid(),smem,stream>>>(p);return cudaGetLastError();
}
extern "C" int ltx25_natten3d(const void* q,const void* k,const void* v,void* out,void* accum,void* lse,
    int batch,int t,int h,int w,int heads,int dim,const int* window,const int* qt,const int* kt,float scale,void* stream) {
    if(dim!=64 || batch<1 || t<window[0] || h<window[1] || w<window[2])return -1;
    int a=qt[0]*qt[1]*qt[2],b=kt[0]*kt[1]*kt[2];auto st=(cudaStream_t)stream;
    if(a==64 && b==64)return forward<64,64>(q,k,v,out,accum,lse,batch,t,h,w,heads,window,qt,kt,scale,st);
    if(a==32 && b==128)return forward<32,128>(q,k,v,out,accum,lse,batch,t,h,w,heads,window,qt,kt,scale,st);
    if(a==64 && b==128)return forward<64,128>(q,k,v,out,accum,lse,batch,t,h,w,heads,window,qt,kt,scale,st);
    return -2;
}
