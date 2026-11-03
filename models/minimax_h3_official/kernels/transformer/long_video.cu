// FL2VA fused-window pointwise operations. No transformer or scheduler fallback.
// Operation boundaries follow the original PyTorch windowed denoising reference.
// Mean reduction preserves the original FP32 reduction/projection order.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {
constexpr int threads=256;
unsigned blocks(int64_t n){return unsigned(std::min<int64_t>((n+threads-1)/threads,65535));}
__global__ void anchor(float* out,const float* clean,const float* noise,int64_t n,float t,float sigma){
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x)
        out[i]=__fadd_rn(__fmul_rn(t,clean[i]),__fmul_rn(sigma,noise[i]));
}
// Adapted reduction order from PyTorch 2.13 Reduce.cuh / MeanOps at
// cf30153c4c131c8164ee7798e5022d810682e2cb (BSD; licenses/PYTORCH-LICENSE).
// Only the FP32 contiguous [rows, 96/32] dim-0 mean is needed here. There is
// no ATen runtime. Preserve four independent accumulators, output-vector
// geometry, warp/CTA splits and multiply-by-reciprocal projection exactly.
struct MeanPlan {int width,height,ctas;bool reduce_y;};
int floor_power2(int n){int p=1;while(p<=n/2)p*=2;return p;}
int ceil_div(int a,int b){return (a+b-1)/b;}
MeanPlan mean_plan(int rows,int cols,int sms,int threads_per_sm){
    const int d0=std::min(128,floor_power2(cols/4)),d1=std::min(128,floor_power2(rows));
    int width=std::min(d0,32),height=std::min(d1,128/width);width=std::min(d0,128/height);
    const bool reduce_y=rows>=std::min(height*16,256);
    int ctas=1;
    if(reduce_y){const int values=ceil_div(rows,height),grid=ceil_div(cols/4,width);
        const int target=sms*(threads_per_sm/(width*height));
        if(values>=256&&grid<=target)
            ctas=std::max(std::min(ceil_div(target,grid),ceil_div(values,16)),ceil_div(values,256));
    }
    return {width,height,ctas,reduce_y};
}
__global__ void means(const float* previous,const float* sample,const float* pred,float* scratch,
                      int rows,int cols,float sigma,MeanPlan plan){
    __shared__ float before[512],current[512];
    const int x=threadIdx.x,y=threadIdx.y;
    const int col=(blockIdx.x*plan.width*(plan.reduce_y?1:plan.height)+x+(plan.reduce_y?0:y*plan.width))*4;
    const int first=plan.reduce_y?y+blockIdx.y*plan.height:0;
    const int stride=plan.reduce_y?plan.height*plan.ctas:1;
    float a[4][4]={},b[4][4]={};
    if(col<cols){
        for(int start=first;start<rows;start+=4*stride){
            #pragma unroll
            for(int k=0;k<4;k++)if(start+k*stride<rows){
                const int64_t offset=int64_t(start+k*stride)*cols+col;
                #pragma unroll
                for(int j=0;j<4;j++){
                    a[k][j]=__fadd_rn(a[k][j],previous[offset+j]);
                    b[k][j]=__fadd_rn(b[k][j],__fadd_rn(sample[offset+j],__fmul_rn(sigma,pred[offset+j])));
                }
            }
        }
    }
    #pragma unroll
    for(int k=1;k<4;k++){
        #pragma unroll
        for(int j=0;j<4;j++){a[0][j]=__fadd_rn(a[0][j],a[k][j]);b[0][j]=__fadd_rn(b[0][j],b[k][j]);}
    }
    const int slot=(y*plan.width+x)*4;
    if(plan.reduce_y){
        #pragma unroll
        for(int j=0;j<4;j++){before[slot+j]=a[0][j];current[slot+j]=b[0][j];}
        for(int offset=plan.height/2;offset;offset/=2){
            __syncthreads();
            if(y<offset){
                #pragma unroll
                for(int j=0;j<4;j++){
                    a[0][j]=__fadd_rn(a[0][j],before[slot+offset*plan.width*4+j]);
                    b[0][j]=__fadd_rn(b[0][j],current[slot+offset*plan.width*4+j]);
                    before[slot+j]=a[0][j];current[slot+j]=b[0][j];
                }
            }
        }
    }
    if(col<cols&&(!plan.reduce_y||y==0)){
        #pragma unroll
        for(int j=0;j<4;j++){
            scratch[int64_t(blockIdx.y)*cols+col+j]=a[0][j];
            scratch[int64_t(plan.ctas)*cols+int64_t(blockIdx.y)*cols+col+j]=b[0][j];
        }
    }
}
__global__ void shift(const float* scratch,float* out,int rows,int cols,float sigma,MeanPlan plan,float factor){
    __shared__ float before[512],current[512];
    const int x=threadIdx.x,y=threadIdx.y,col=(blockIdx.x*plan.width+x)*4;
    const int slot=(y*plan.width+x)*4;float a[4]={},b[4]={};
    if(col<cols){
        for(int i=y;i<plan.ctas;i+=plan.height){
            #pragma unroll
            for(int j=0;j<4;j++){
                a[j]=__fadd_rn(a[j],scratch[int64_t(i)*cols+col+j]);
                b[j]=__fadd_rn(b[j],scratch[int64_t(plan.ctas+i)*cols+col+j]);
            }
        }
    }
    #pragma unroll
    for(int j=0;j<4;j++){before[slot+j]=a[j];current[slot+j]=b[j];}
    for(int offset=plan.height/2;offset;offset/=2){
        __syncthreads();
        if(y<offset){
            #pragma unroll
            for(int j=0;j<4;j++){
                a[j]=__fadd_rn(a[j],before[slot+offset*plan.width*4+j]);
                b[j]=__fadd_rn(b[j],current[slot+offset*plan.width*4+j]);
                before[slot+j]=a[j];current[slot+j]=b[j];
            }
        }
    }
    if(y==0&&col<cols){
        #pragma unroll
        for(int j=0;j<4;j++)out[col+j]=__fdiv_rn(__fsub_rn(__fmul_rn(a[j],factor),__fmul_rn(b[j],factor)),sigma);
    }
}
__global__ void accumulate(const float* sample,const float* pred,const float* color,
                           const float* weights,float* sum,float* weight_sum,float* x0,
                           int64_t rows,int cols,int rows_per_weight,float sigma){
    const int64_t n=rows*cols;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x){
        const int64_t row=i/cols;const int col=i%cols;const float w=weights[row/rows_per_weight];
        const float v=color?__fadd_rn(pred[i],color[col]):pred[i];
        sum[i]=__fadd_rn(sum[i],__fmul_rn(w,v));
        if(col==0)weight_sum[row]=__fadd_rn(weight_sum[row],w);
        if(x0)x0[i]=__fadd_rn(sample[i],__fmul_rn(sigma,v));
    }
}
__global__ void blend(const float* sample,float* velocity,const float* weights,float* x0,
                      int64_t rows,int cols,float sigma){
    const int64_t n=rows*cols;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x){
        const float v=__fdiv_rn(velocity[i],fmaxf(weights[i/cols],1e-12f));
        velocity[i]=v;
        if(x0)x0[i]=__fadd_rn(sample[i],__fmul_rn(sigma,v));
    }
}
bool valid_rows(int64_t rows,int cols){return rows>0&&rows<=INT32_MAX&&(cols==96||cols==32);}
}
extern "C" int h3_long_anchor(float* out,const float* clean,const float* noise,int64_t n,float t,cudaStream_t stream){
    if(!out||!clean||!noise||n<=0||n>INT32_MAX||!std::isfinite(t)||t<0.f||t>1.f)return cudaErrorInvalidValue;
    anchor<<<blocks(n),threads,0,stream>>>(out,clean,noise,n,t,1.f-t);return cudaGetLastError();
}
extern "C" size_t h3_long_color_workspace(int64_t rows,int cols){
    return valid_rows(rows,cols)?size_t((rows+127)/128)*cols*2*sizeof(float):0;
}
extern "C" int h3_long_color_shift(const float* previous,const float* sample,const float* pred,
                                    float* scratch,size_t bytes,float* out,int64_t rows,int cols,
                                    float sigma,cudaStream_t stream){
    const size_t required=h3_long_color_workspace(rows,cols);
    if(!previous||!sample||!pred||!scratch||!out||!required||bytes<required||
       !std::isfinite(sigma)||sigma<=0.f||sigma>1.f)return cudaErrorInvalidValue;
    int device=0,sms=0,threads_per_sm=0;
    auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return status;
    status=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);if(status!=cudaSuccess)return status;
    status=cudaDeviceGetAttribute(&threads_per_sm,cudaDevAttrMaxThreadsPerMultiProcessor,device);if(status!=cudaSuccess)return status;
    const MeanPlan plan=mean_plan(int(rows),cols,sms,threads_per_sm);
    if(size_t(plan.ctas)*cols*2*sizeof(float)>bytes)return cudaErrorInvalidValue;
    const int grid=ceil_div(cols/4,plan.width*(plan.reduce_y?1:plan.height));
    means<<<dim3(grid,plan.ctas),dim3(plan.width,plan.height),0,stream>>>(previous,sample,pred,scratch,int(rows),cols,sigma,plan);
    status=cudaGetLastError();if(status!=cudaSuccess)return status;
    shift<<<ceil_div(cols/4,plan.width),dim3(plan.width,plan.height),0,stream>>>(scratch,out,int(rows),cols,sigma,plan,1.f/float(rows));
    return cudaGetLastError();
}
extern "C" int h3_long_accumulate(const float* sample,const float* pred,const float* color,const float* weights,
                                   float* sum,float* weight_sum,float* x0,int64_t rows,int cols,
                                   int rows_per_weight,float sigma,cudaStream_t stream){
    if(!pred||!weights||!sum||!weight_sum||(x0&&!sample)||!valid_rows(rows,cols)||rows_per_weight<1||
       rows%rows_per_weight||!std::isfinite(sigma)||sigma<=0.f||sigma>1.f)return cudaErrorInvalidValue;
    accumulate<<<blocks(rows*cols),threads,0,stream>>>(sample,pred,color,weights,sum,weight_sum,x0,rows,cols,rows_per_weight,sigma);
    return cudaGetLastError();
}
extern "C" int h3_long_blend(const float* sample,float* velocity,const float* weights,float* x0,
                              int64_t rows,int cols,float sigma,cudaStream_t stream){
    if(!velocity||!weights||(x0&&!sample)||!valid_rows(rows,cols)||!std::isfinite(sigma)||sigma<=0.f||sigma>1.f)
        return cudaErrorInvalidValue;
    blend<<<blocks(rows*cols),threads,0,stream>>>(sample,velocity,weights,x0,rows,cols,sigma);return cudaGetLastError();
}
