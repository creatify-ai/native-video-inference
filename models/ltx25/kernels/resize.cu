// PyTorch cf30153c4c131c81 UpSampleBilinear2d.cu's RGB (<16-channel)
// align_corners=false CUDA path. BSD license: licenses/PYTORCH-LICENSE.
#include <cuda_runtime.h>
#include <cstdint>
namespace {
__global__ void bilinear(const uint8_t* source,float* output,int ih,int iw,int oh,int ow,float rh,float rw){
    int index=threadIdx.x+blockIdx.x*blockDim.x;if(index>=oh*ow)return;
    int w2=index%ow,h2=index/ow;
    float h1r=rh*(h2+0.5f)-0.5f;h1r=h1r<0?0:h1r;
    float w1r=rw*(w2+0.5f)-0.5f;w1r=w1r<0?0:w1r;
    int h1=h1r,w1=w1r,hp=h1<ih-1?1:0,wp=w1<iw-1?1:0;
    float hl=h1r-h1,wl=w1r-w1,h0=1-hl,w0=1-wl;
    for(int c=0;c<3;c++){
        float value=h0*(w0*float(source[(h1*iw+w1)*3+c])+wl*float(source[(h1*iw+w1+wp)*3+c]))
            +hl*(w0*float(source[((h1+hp)*iw+w1)*3+c])+wl*float(source[((h1+hp)*iw+w1+wp)*3+c]));
        output[index*3+c]=value;
    }
}
}
extern "C" int ltx25_bilinear_rgb(const uint8_t* source,float* output,int ih,int iw,int oh,int ow,cudaStream_t stream){
    bilinear<<<(oh*ow+1023)/1024,1024,0,stream>>>(source,output,ih,iw,oh,ow,float(ih)/oh,float(iw)/ow);
    return int(cudaGetLastError());
}
