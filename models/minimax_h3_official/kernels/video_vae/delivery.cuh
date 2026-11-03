// Native port of the pinned PyTorch CUDA bicubic delivery arithmetic.
// Reference: ATen/native/cuda/UpSample.cuh, cf30153c4c131c8164ee7798e5022d810682e2cb.
// RGB is resized for delivery only; model latents are never altered here.
namespace {
__device__ float resize_cubic_aa(float x){
 x=fabsf(x);
 if(x<1.f)return (1.5f*x-2.5f)*x*x+1.f;
 if(x<2.f)return (((x-5.f)*x+8.f)*x-4.f)*-.5f;
 return 0.f;
}
__device__ void resize_coeff(float t,float*wt){
 float x=t+1.f;wt[0]=((-.75f*x+3.75f)*x-6.f)*x+3.f;
 wt[1]=(1.25f*t-2.25f)*t*t+1.f;
 x=1.f-t;wt[2]=(1.25f*x-2.25f)*x*x+1.f;
 x+=1.f;wt[3]=((-.75f*x+3.75f)*x-6.f)*x+3.f;
}
__device__ void resize_weights(int old,int dest,int coord,bool aa,int&first,int&size,float*weights){
 float scale=float(old)/float(dest);
 if(!aa){float real=scale*(coord+.5f)-.5f;int base=int(floorf(real));first=base-1;size=4;resize_coeff(real-base,weights);return;}
 float center=scale*(coord+.5f),support=2.f*fmaxf(scale,1.f);
 first=max(int(center-support+.5f),0);size=min(int(center+support+.5f),old)-first;
 float offset=fmaf(-scale,coord+.5f,float(first)),inverse=scale>=1.f?1.f/scale:1.f,total=0.f;
 for(int j=0;j<size;++j){weights[j]=resize_cubic_aa((float(j)+offset+.5f)*inverse);total+=weights[j];}
 for(int j=0;j<size;++j)weights[j]/=total;
}
__device__ float resize_sum4(float x0,float x1,float x2,float x3,const float*w){
 return fmaf(x3,w[3],fmaf(x2,w[2],fmaf(x0,w[0],x1*w[1])));
}
__global__ void resize_resize_x(const unsigned char*in,float*out,int n,int h,int w,int ow,bool aa){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i>=int64_t(n)*h*ow*3)return;
 int c=i%3,x=(i/3)%ow;int64_t row=i/(ow*3);float weights[64];int first,size;
 resize_weights(w,ow,x,aa,first,size,weights);float value=0.f;
 if(!aa){float values[4];for(int j=0;j<4;++j)values[j]=float(in[(row*w+max(0,min(w-1,first+j)))*3+c]);value=resize_sum4(values[0],values[1],values[2],values[3],weights);}
 else for(int j=0;j<size;++j){float v=float(in[(row*w+first+j)*3+c]);value=j==0?v*weights[j]:fmaf(v,weights[j],value);}
 out[i]=value;
}
__global__ void resize_resize_y(const float*in,unsigned char*out,int n,int h,int w,int oh,bool aa){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i>=int64_t(n)*oh*w*3)return;
 int c=i%3,x=(i/3)%w,y=(i/(w*3))%oh,b=i/(int64_t(oh)*w*3);float weights[64];int first,size;
 resize_weights(h,oh,y,aa,first,size,weights);float value=0.f;
 if(!aa){float values[4];for(int j=0;j<4;++j)values[j]=in[((int64_t(b)*h+max(0,min(h-1,first+j)))*w+x)*3+c];value=resize_sum4(values[0],values[1],values[2],values[3],weights);}
 else for(int j=0;j<size;++j){float v=in[((int64_t(b)*h+first+j)*w+x)*3+c];value=j==0?v*weights[j]:fmaf(v,weights[j],value);}
 out[i]=static_cast<unsigned char>(max(0,min(255,__float2int_rn(value))));
}
}
extern "C" int h3vg_delivery_resize(const unsigned char*in,float*scratch,unsigned char*out,
 int n,int h,int w,int oh,int ow,void*stream){
 if(n<1||n>128||h<1||w<1||oh<1||ow<1||h>oh*15||w>ow*15)return 1;
 bool aa=oh<h||ow<w;
 resize_resize_x<<<(int64_t(n)*h*ow*3+255)/256,256,0,(cudaStream_t)stream>>>(in,scratch,n,h,w,ow,aa);
 resize_resize_y<<<(int64_t(n)*oh*ow*3+255)/256,256,0,(cudaStream_t)stream>>>(scratch,out,n,h,ow,oh,aa);
 return int(cudaGetLastError());
}
