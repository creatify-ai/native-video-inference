// H3 keeps its original budget and ABI; backend arithmetic belongs to core.
#include "../common/conv.h"
#include "../../../../core_infra/kernels/conv.cpp"
extern "C" int h3_conv_create(void** out) { return infra_conv_create(out,640,size_t(2)<<30); }
extern "C" int h3_conv_forward(void* c,const H3ConvSpec* s,const void* x,const void* w,void* y,void* stream) { return infra_conv_forward(c,s,x,w,y,stream); }
extern "C" const char* h3_conv_error() { return infra_conv_error(); }
extern "C" const char* h3_conv_plan_tag(void* c) { return infra_conv_plan_tag(c); }
extern "C" uint64_t h3_conv_workspace_bytes(void* c) { return infra_conv_workspace_bytes(c); }
extern "C" int h3_conv_destroy(void* c) { return infra_conv_destroy(c); }
extern "C" int h3_conv_cache_stats(void* c,uint64_t* out) { return infra_conv_cache_stats(c,out); }
