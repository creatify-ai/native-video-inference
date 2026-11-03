#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

struct Layout { int64_t shape[3], a_stride[3], b_stride[3]; };

// Each original TensorIterator primitive remains a separate launch/rounding.
// 0 copy, 1 multiply, 2 add, 3 sin, 4 square, 5 clamp[-1,1].
__global__ void pointwise(int op, const float* a, const float* b, float* output,
                          int64_t count, Layout layout, float sa, float sb) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t position = i, ai = 0, bi = 0;
        #pragma unroll
        for (int d = 2; d >= 0; --d) {
            int64_t coordinate = position % layout.shape[d];
            position /= layout.shape[d];
            ai += coordinate * layout.a_stride[d];
            bi += coordinate * layout.b_stride[d];
        }
        float av = a ? a[ai] : sa;
        float bv = b ? b[bi] : sb;
        float value;
        if (op == 0) value = av;
        else if (op == 1) value = __fmul_rn(av, bv);
        else if (op == 2) value = __fadd_rn(av, bv);
        else if (op == 3) value = sinf(av);
        else if (op == 4) value = __fmul_rn(av, av);
        else value = isnan(av) ? av : (av < -1.0f ? -1.0f : (av > 1.0f ? 1.0f : av));
        output[i] = value;
    }
}
extern "C" int h3_audio_pointwise(int op, const void* a, const void* b, void* output,
    int64_t count, const int64_t* shape, const int64_t* as, const int64_t* bs,
    float sa, float sb, void* stream) {
    Layout layout;
    for (int d = 0; d < 3; ++d) { layout.shape[d]=shape[d]; layout.a_stride[d]=as[d]; layout.b_stride[d]=bs[d]; }
    int blocks = int((count + 255) / 256);
    if (blocks > 65535) blocks = 65535;
    pointwise<<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(op,
        static_cast<const float*>(a),static_cast<const float*>(b),static_cast<float*>(output),count,layout,sa,sb);
    return cudaGetLastError();
}

__global__ void replication_pad(const float* input, float* output, int64_t count,
    int input_length, int output_length, int left) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t row = i / output_length;
        int column = int(i % output_length) - left;
        column = max(0,min(input_length-1,column));
        output[i] = input[row * input_length + column];
    }
}
extern "C" int h3_audio_pad(const void* input, void* output, int64_t count,
    int input_length, int output_length, int left, void* stream) {
    replication_pad<<<int((count+255)/256),256,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(input),static_cast<float*>(output),count,input_length,output_length,left);
    return cudaGetLastError();
}

// Direct scalar specialization of pinned PyTorch's
// conv_depthwise2d_forward_kernel_generic<float,int32_t>, H=kernelH=1.
// Increasing k order and the original multiply-add contraction are preserved.
__global__ void depthwise(const float* input, const float* weight, const float* bias,
    float* output, int count, int channels, int input_length, int output_length,
    int kernel, int stride, int padding, int dilation) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        int column = i % output_length;
        int row = i / output_length;
        int channel = row % channels;
        int start = -padding + column * stride;
        int lo = start < 0 ? (-start + dilation - 1) / dilation : 0;
        int over = start + (kernel - 1) * dilation - input_length + 1;
        int hi = over >= 0 ? kernel - (over + dilation - 1) / dilation : kernel;
        float sum = bias ? bias[channel] : 0.0f;
        for (int k = lo; k < hi; ++k)
            sum = __fmaf_rn(weight[channel * kernel + k], input[row * input_length + start + k * dilation], sum);
        output[i] = sum;
    }
}
extern "C" int h3_audio_depthwise(const void* input, const void* weight, const void* bias,
    void* output, int count, int channels, int input_length, int output_length,
    int kernel, int stride, int padding, int dilation, void* stream) {
    depthwise<<<(count+1023)/1024,1024,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(input),static_cast<const float*>(weight),static_cast<const float*>(bias),
        static_cast<float*>(output),count,channels,input_length,output_length,kernel,stride,padding,dilation);
    return cudaGetLastError();
}

template<typename T> __global__ void channel_bias(T* output,const T* bias,int64_t count,int64_t channels,int64_t spatial) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x)
        output[i]=output[i]+bias[(i/spatial)%channels];
}
extern "C" int h3_conv_bias_add(void* output,const void* bias,int64_t count,int64_t channels,int64_t spatial,int dtype,void* stream) {
    int blocks=int((count+255)/256);if(blocks>65535)blocks=65535;
    if(dtype==0)channel_bias<<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<float*>(output),static_cast<const float*>(bias),count,channels,spatial);
    else if(dtype==1)channel_bias<<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<__half*>(output),static_cast<const __half*>(bias),count,channels,spatial);
    else if(dtype==2)channel_bias<<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<__nv_bfloat16*>(output),static_cast<const __nv_bfloat16*>(bias),count,channels,spatial);
    else return -1;
    return cudaGetLastError();
}

// Read-only output telemetry; no changes to decoder arithmetic.
__global__ void audio_health(const float* input,double* output,int64_t count){
    __shared__ double sums[256],peaks[256];double sum=0,peak=0;
    for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<count;i+=int64_t(gridDim.x)*256){
        double value=double(input[i]);sum=__dadd_rn(sum,__dmul_rn(value,value));peak=fmax(peak,fabs(value));
    }
    sums[threadIdx.x]=sum;peaks[threadIdx.x]=peak;__syncthreads();
    for(int stride=128;stride;stride/=2){if(threadIdx.x<stride){
        sums[threadIdx.x]=__dadd_rn(sums[threadIdx.x],sums[threadIdx.x+stride]);
        peaks[threadIdx.x]=fmax(peaks[threadIdx.x],peaks[threadIdx.x+stride]);
    }__syncthreads();}
    if(threadIdx.x==0){output[2*blockIdx.x]=sums[0];output[2*blockIdx.x+1]=peaks[0];}
}
extern "C" int h3_audio_health(const float* input,double* output,int64_t count,void* stream){
    if(count<=0)return cudaErrorInvalidValue;
    audio_health<<<256,256,0,static_cast<cudaStream_t>(stream)>>>(input,output,count);return cudaGetLastError();
}
