#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// SGLang reverses ImageNet normalization with torchvision Normalize on the
// FP16 decoder output, then promotes to FP32 and truncates the uint8 cast.
// Keep the two FP16 roundings: an FP32 multiply/add is not equivalent.
__device__ inline uint8_t h3_public_rgb_pixel(__half value,int channel) {
    double mean=channel==0?.485:(channel==1?.456:.406);
    double std=channel==0?.229:(channel==1?.224:.225);
    __half shifted=__hsub(value,__float2half_rn(float(-mean/std)));
    __half normalized=__hdiv(shifted,__float2half_rn(float(1./std)));
    float pixel=fminf(1.f,fmaxf(0.f,__half2float(normalized)));
    return uint8_t(__float2uint_rz(__fmul_rn(pixel,255.f)));
}
