// Exact BF16-bit relayout only: no arithmetic or conversion of tensor values.
#include <cuda_runtime.h>
#include <stdint.h>
#include <limits.h>

namespace {
constexpr int threads = 256;
// Eight adjacent BF16 values per vector. All head boundaries are 128 elements;
// select the peer via grid.y so no per-element variable row division is needed.
template<int World>
__global__ void pack_vectors(const uint4* q,const uint4* k,const uint4* v,uint4* out,unsigned rows){
    constexpr unsigned local=56/World, width=local*16;
    const unsigned peer=blockIdx.y, n=rows*width;
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
        const unsigned row=i/width, head=(i/16)%local, dim=i%16;
        const uint64_t source=(uint64_t(row)*56+peer*local+head)*16+dim;
        const uint64_t target=((uint64_t(peer)*rows+row)*local+head)*48+dim;
        out[target]=q[source];out[target+16]=k[source];out[target+32]=v[source];
    }
}
__global__ void unpack_vectors(const uint4* input,uint4* q,uint4* k,uint4* v,unsigned n){
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
        const uint64_t base=uint64_t(i/16)*48+i%16;
        q[i]=input[base];k[i]=input[base+16];v[i]=input[base+32];
    }
}
template<int World>
__global__ void merge_vectors(const uint4* input,uint4* output,unsigned rows){
    constexpr unsigned local=56/World;
    const unsigned n=rows*56*16;
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
        const unsigned head=(i/16)%56,row=i/(56*16),dim=i%16;
        const uint64_t source=((uint64_t(head/local)*rows+row)*local+head%local)*16+dim;
        output[i]=input[source];
    }
}
// Destination-major [world, rows_local, heads_local, 3, D].
__global__ void pack_qkv(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                         uint16_t* out, int64_t rows, int heads, int world) {
    const int local_heads = heads / world;
    const int64_t n = rows * heads * 128 * 3;
    for (int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x; i<n;
         i += int64_t(gridDim.x)*blockDim.x) {
        int64_t x=i; const int d=x%128; x/=128;
        const int which=x%3; x/=3; const int h=x%local_heads; x/=local_heads;
        const int64_t row=x%rows; const int peer=x/rows;
        const int64_t source=(row*heads+peer*local_heads+h)*128+d;
        out[i]=(which==0?q:which==1?k:v)[source];
    }
}
// Rank-concatenated receive is [rows_full, heads_local, 3, D].
__global__ void unpack_qkv(const uint16_t* packed, uint16_t* q, uint16_t* k,
                           uint16_t* v, int64_t elements) {
    for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; i<elements;
         i+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t base=(i/128)*384+i%128;
        q[i]=packed[base]; k[i]=packed[base+128]; v[i]=packed[base+256];
    }
}
// Receive [world, rows_local, heads_local, D] -> [rows_local, heads, D].
__global__ void merge_heads(const uint16_t* input, uint16_t* output,
                            int64_t rows, int heads, int world) {
    const int local_heads=heads/world;
    const int64_t n=rows*heads*128;
    for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; i<n;
         i+=int64_t(gridDim.x)*blockDim.x) {
        const int d=i%128; const int h=(i/128)%heads; const int64_t row=i/(heads*128);
        output[i]=input[((int64_t(h/local_heads)*rows+row)*local_heads+h%local_heads)*128+d];
    }
}
// [rows,56,128] <-> [rows,group_heads,128]. Heads are independent;
// preserve every BF16 bit, including for offsets above signed 32-bit range.
__global__ void copy_heads(const uint16_t* input, uint16_t* output,
                           int64_t rows, int first, int heads, bool scatter) {
    const int64_t n=rows*heads*128;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; i<n;
        i+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t row=i/(heads*128);
        const int64_t full=(row*56+first)*128+i%(heads*128);
        if(scatter) output[full]=input[i];
        else output[i]=input[full];
    }
}
int grid(int64_t n) { return int(((n+threads-1)/threads)>65535?65535:(n+threads-1)/threads); }
bool valid(int64_t rows,int heads,int world) {
    return rows>0 && rows<=524288 && heads==56 && (world==2 || world==4 || world==8);
}
}
extern "C" int h3_attention_copy_heads(const void* input,void* output,int64_t rows,
                                        int first,int heads,int scatter,void* stream) {
    if(!input||!output||input==output||rows<=0||rows>524288||first<0||heads<=0||
       first>56-heads||(heads!=7&&heads!=14&&heads!=28&&heads!=56)||(scatter!=0&&scatter!=1))
        return cudaErrorInvalidValue;
    copy_heads<<<grid(rows*heads*128),threads,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint16_t*>(input),static_cast<uint16_t*>(output),rows,first,heads,scatter!=0);
    return cudaGetLastError();
}
extern "C" int h3_cp_pack_qkv(const void* q,const void* k,const void* v,void* output,
                               int64_t rows,int heads,int world,void* stream) {
    if(!q||!k||!v||!output||!valid(rows,heads,world))return cudaErrorInvalidValue;
    if(((uintptr_t(q)|uintptr_t(k)|uintptr_t(v)|uintptr_t(output))&15u)==0){
        const dim3 blocks(grid(rows*(56/world)*16),unsigned(world));
        const auto s=static_cast<cudaStream_t>(stream);
        if(world==2)pack_vectors<2><<<blocks,threads,0,s>>>((const uint4*)q,(const uint4*)k,(const uint4*)v,(uint4*)output,unsigned(rows));
        else if(world==4)pack_vectors<4><<<blocks,threads,0,s>>>((const uint4*)q,(const uint4*)k,(const uint4*)v,(uint4*)output,unsigned(rows));
        else pack_vectors<8><<<blocks,threads,0,s>>>((const uint4*)q,(const uint4*)k,(const uint4*)v,(uint4*)output,unsigned(rows));
        return cudaGetLastError();
    }
    pack_qkv<<<grid(rows*heads*128*3),threads,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint16_t*>(q),static_cast<const uint16_t*>(k),static_cast<const uint16_t*>(v),
        static_cast<uint16_t*>(output),rows,heads,world);
    return cudaGetLastError();
}
extern "C" int h3_cp_unpack_qkv(const void* input,void* q,void* k,void* v,
                                 int64_t rows,int heads,void* stream) {
    if(!input||!q||!k||!v||rows<=0||rows>524288||(heads!=28&&heads!=14&&heads!=7))return cudaErrorInvalidValue;
    const int64_t n=rows*heads*128;
    if(((uintptr_t(input)|uintptr_t(q)|uintptr_t(k)|uintptr_t(v))&15u)==0){
        unpack_vectors<<<grid(n/8),threads,0,static_cast<cudaStream_t>(stream)>>>(
            (const uint4*)input,(uint4*)q,(uint4*)k,(uint4*)v,unsigned(n/8));
        return cudaGetLastError();
    }
    unpack_qkv<<<grid(n),threads,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint16_t*>(input),static_cast<uint16_t*>(q),static_cast<uint16_t*>(k),
        static_cast<uint16_t*>(v),n);
    return cudaGetLastError();
}
extern "C" int h3_cp_merge_heads(const void* input,void* output,int64_t rows,
                                  int heads,int world,void* stream) {
    if(!input||!output||!valid(rows,heads,world))return cudaErrorInvalidValue;
    if(((uintptr_t(input)|uintptr_t(output))&15u)==0){
        const int blocks=grid(rows*56*16);const auto s=static_cast<cudaStream_t>(stream);
        if(world==2)merge_vectors<2><<<blocks,threads,0,s>>>((const uint4*)input,(uint4*)output,unsigned(rows));
        else if(world==4)merge_vectors<4><<<blocks,threads,0,s>>>((const uint4*)input,(uint4*)output,unsigned(rows));
        else merge_vectors<8><<<blocks,threads,0,s>>>((const uint4*)input,(uint4*)output,unsigned(rows));
        return cudaGetLastError();
    }
    merge_heads<<<grid(rows*heads*128),threads,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint16_t*>(input),static_cast<uint16_t*>(output),rows,heads,world);
    return cudaGetLastError();
}

// Direct exchange: gather from peers' device memory (CUDA IPC views) straight into
// the layouts attention and the output projection consume. Exact BF16 relayout only.
// The qkv source is the destination-major staging written by h3_cp_pack_qkv, so each
// peer reads one contiguous block; the attention output is read in place.
namespace {
struct Peers { const uint4* p[8]; };
struct Starts { unsigned s[9]; };
// [S_full, heads_local, 128] for my heads: row r lives on the peer owning r, at its
// local row r - start[p], heads rank*local .. rank*local+local-1 (contiguous 16*local uint4).
// [rows_local, 56, 128] from every peer's [S_full, heads_local, 128] attention output.
template<int World>
__global__ void pull_out_vectors(Peers attn,uint4* out,unsigned start,unsigned rows){
    constexpr unsigned local=56/World;
    const unsigned n=rows*56*16;
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
        const unsigned row=i/(56*16), head=(i/16)%56, dim=i%16;
        const unsigned p=head/local, h=head%local;
        out[i]=attn.p[p][(uint64_t(start+row)*local+h)*16+dim];
    }
}
bool aligned16(const void* p){return (uintptr_t(p)&15u)==0;}
}
extern "C" int h3_cp_pull_out(const void* const* ap,int world,int64_t start,int64_t rows,void* out,void* stream){
    if(!ap||!out||rows<=0||rows>524288||start<0||start+rows>524288||(world!=2&&world!=4&&world!=8)||!aligned16(out))
        return cudaErrorInvalidValue;
    Peers attn{};
    for(int p=0;p<world;++p){if(!ap[p]||!aligned16(ap[p]))return cudaErrorInvalidValue;attn.p[p]=(const uint4*)ap[p];}
    const int blocks=grid(rows*56*16);const auto s=static_cast<cudaStream_t>(stream);
    if(world==2)pull_out_vectors<2><<<blocks,threads,0,s>>>(attn,(uint4*)out,unsigned(start),unsigned(rows));
    else if(world==4)pull_out_vectors<4><<<blocks,threads,0,s>>>(attn,(uint4*)out,unsigned(start),unsigned(rows));
    else pull_out_vectors<8><<<blocks,threads,0,s>>>(attn,(uint4*)out,unsigned(start),unsigned(rows));
    return cudaGetLastError();
}

// Contiguous-source variant: each rank first packs its own q/k/v into the
// destination-major staging layout [world][rows_local][heads_local][3][128]
// (h3_cp_pack_qkv), so the slice a peer needs is one contiguous block. Remote
// reads stream sequentially; local writes split into three 256-byte runs.
template<int World>
__global__ void pull_qkv_packed(Peers send,uint4* oq,uint4* ok,uint4* ov,Starts st,unsigned rank,unsigned max_rows){
    constexpr unsigned local=56/World, width=local*48;   // uint4 per packed row
    const unsigned n=max_rows*World*width;
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
        const unsigned j=i/width, col=i%width;
        const unsigned p=j%World, lrow=j/World, count=st.s[p+1]-st.s[p];
        if(lrow>=count) continue;
        const unsigned row=st.s[p]+lrow, h=col/48, which=(col/16)%3, d=col%16;
        const uint4 value=send.p[p][(uint64_t(rank)*count+lrow)*width+col];
        uint4* dst=which==0?oq:which==1?ok:ov;
        dst[(uint64_t(row)*local+h)*16+d]=value;
    }
}
extern "C" int h3_cp_pull_qkv_packed(const void* const* sp,const uint32_t* starts,int world,int rank,int64_t seq,
                                      void* oq,void* ok,void* ov,void* stream){
    if(!sp||!starts||!oq||!ok||!ov||seq<=0||seq>524288||(world!=2&&world!=4&&world!=8)||rank<0||rank>=world)
        return cudaErrorInvalidValue;
    if(!aligned16(oq)||!aligned16(ok)||!aligned16(ov))return cudaErrorInvalidValue;
    Peers send{};Starts st{};
    for(int p=0;p<world;++p){if(!sp[p]||!aligned16(sp[p]))return cudaErrorInvalidValue;send.p[p]=(const uint4*)sp[p];}
    for(int p=0;p<=world;++p)st.s[p]=starts[p];
    if(st.s[0]!=0||st.s[world]!=unsigned(seq))return cudaErrorInvalidValue;
    unsigned max_rows=0;for(int p=0;p<world;++p){if(st.s[p+1]<st.s[p])return cudaErrorInvalidValue;max_rows=st.s[p+1]-st.s[p]>max_rows?st.s[p+1]-st.s[p]:max_rows;}
    const int blocks=grid(int64_t(max_rows)*world*(56/world)*48);const auto s=static_cast<cudaStream_t>(stream);
    if(world==2)pull_qkv_packed<2><<<blocks,threads,0,s>>>(send,(uint4*)oq,(uint4*)ok,(uint4*)ov,st,unsigned(rank),max_rows);
    else if(world==4)pull_qkv_packed<4><<<blocks,threads,0,s>>>(send,(uint4*)oq,(uint4*)ok,(uint4*)ov,st,unsigned(rank),max_rows);
    else pull_qkv_packed<8><<<blocks,threads,0,s>>>(send,(uint4*)oq,(uint4*)ok,(uint4*)ov,st,unsigned(rank),max_rows);
    return cudaGetLastError();
}

