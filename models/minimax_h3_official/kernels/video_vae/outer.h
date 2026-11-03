#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// All pointers are device pointers; caller owns storage and stream lifetime.
// Fixed case-01: 768x768 RGB keyframes, 24x102x48x48 video latents.
int h3v_encode_prepare(const uint8_t* rgb, float* pixels, int channels_last, void* stream);
int h3v_extract_f32(const float* input, float* tile, int channels, int frames,
    int height, int width, int t0, int y0, int x0, int tile_t, int tile_h, int tile_w,
    int channels_last, void* stream);
// tiles: contiguous [16,C,T,K,K], row-major tile order. dtype 0=FP32, 1=FP16.
// K=16 for moments (48x1), K=256 for decoded pixels (3x28).
int h3v_stitch(const void* tiles, void* output, int channels, int frames,
    int tile_size, int dtype, void* stream);
int h3v_decode_denorm(const float* input, const float* mean, const float* std,
    float* output, void* stream);
// moments [48,1,48,48], noise [24,1,48,48], mean/std [24], result FP32.
int h3v_posterior(const float* moments, const float* noise, const float* mean,
    const float* std, float* output, void* stream);
// clip FP16 [3,28,768,768], overlap FP16 [3,5,768,768], result [3,345,768,768].
// Invoke sequentially for chunk_index 0..19; overlap is initialized by chunk 0.
int h3v_temporal_merge(const void* clip, void* overlap, void* output,
    int chunk_index, void* stream);
int h3v_pt8(const void* input, uint8_t* output, void* stream);
// Atomically add the number of FP16 NaN/Inf elements to caller-zeroed counter.
int h3v_half_nonfinite(const void* input, uint32_t* counter, int64_t count, void* stream);
#ifdef __cplusplus
}
#endif
