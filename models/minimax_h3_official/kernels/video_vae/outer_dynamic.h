#pragma once
#include <stdint.h>
extern "C" {
int h3vg_encode_prepare(const uint8_t*,float*,int height,int width,void*);
int h3vg_extract(const float*,float*,int channels,int frames,int height,int width,
    int t0,int y0,int x0,int tile_t,int tile_h,int tile_w,void*);
// Immutable row-major tile array. Starts/overlaps are caller-owned device int32 arrays.
int h3vg_stitch(const void*,void*,int channels,int frames,int tile_size,int height,int width,
    int ny,int nx,const int* y_starts,const int* x_starts,const int* y_overlaps,const int* x_overlaps,int dtype,void*);
int h3vg_denorm(const float*,const float*,const float*,float*,int64_t spatial_temporal,void*);
int h3vg_posterior(const float*,const float*,const float*,const float*,float*,int64_t spatial,void*);
int h3vg_temporal_merge(const void*,void*,void*,int chunk,int chunks,int height,int width,int output_frames,void*);
int h3vg_pt8(const void*,uint8_t*,int height,int width,int frames,void*);
}
