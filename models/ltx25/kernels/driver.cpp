// CUDA driver symbols resolved at runtime: CPU builds need no libcuda stub.
#include <cstdint>
#include <dlfcn.h>
namespace {
void* driver() { static void* handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL); return handle; }
template<class T> T symbol(const char* name) { auto h=driver();return h?reinterpret_cast<T>(dlsym(h,name)):nullptr; }
}
#define DRIVER(name,decl,args) extern "C" int name decl { \
    static auto driver_entry=symbol<decltype(&name)>(#name); return driver_entry?driver_entry args:999; }
DRIVER(cuModuleLoadData,(void** module,const void* image),(module,image))
DRIVER(cuModuleUnload,(void* module),(module))
DRIVER(cuModuleGetFunction,(void** function,void* module,const char* name),(function,module,name))
DRIVER(cuFuncSetAttribute,(void* function,int attribute,int value),(function,attribute,value))
DRIVER(cuGetErrorString,(int code,const char** text),(code,text))
DRIVER(cuCtxGetCurrent,(void** context),(context))
DRIVER(cuLaunchKernel,(void* function,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,
    unsigned shared,void* stream,void** params,void** extra),(function,gx,gy,gz,bx,by,bz,shared,stream,params,extra))
DRIVER(cuTensorMapEncodeTiled,(void* map,unsigned dtype,unsigned rank,void* address,const uint64_t* dims,
    const uint64_t* strides,const uint32_t* boxes,const uint32_t* elements,unsigned interleave,unsigned swizzle,
    unsigned promotion,unsigned fill),(map,dtype,rank,address,dims,strides,boxes,elements,interleave,swizzle,promotion,fill))
