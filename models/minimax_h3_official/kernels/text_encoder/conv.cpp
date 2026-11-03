// Adapted from PyTorch cf30153c4c131c8164ee7798e5022d810682e2cb
// aten/src/ATen/native/cudnn/Conv_v8.cpp and ATen/cudnn/Descriptors.h.
// See ../vendor_reference/PYTORCH-LICENSE and ../NOTICE.md.
#include "conv.h"
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_get_plan.h>
#include <array>
#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
thread_local std::string error;
void check(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(status));
}
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Context {
    cudnnHandle_t handle{};
    std::unordered_map<std::string, std::unique_ptr<cudnn_frontend::ExecutionPlan>> plans;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    std::string last_tag;
    Context() { check(cudnnCreate(&handle)); }
    ~Context() {
        plans.clear();
        if (workspace) cudaFree(workspace);
        if (handle) cudnnDestroy(handle);
    }
    void reserve(size_t count) {
        if (count <= workspace_bytes) return;
        void* next = nullptr;
        cuda_check(cudaMalloc(&next, count));
        if (workspace) cuda_check(cudaFree(workspace));
        workspace = next;
        workspace_bytes = count;
    }
};
uint8_t alignment(const void* pointer) {
    uint8_t result = 1;
    for (; result < 32; result *= 2)
        if (reinterpret_cast<uintptr_t>(pointer) % (result * 2)) return result;
    return result;
}
cudnn_frontend::Tensor tensor(const H3ConvSpec& spec, const int64_t* dim,
                             const int64_t* original_stride, int64_t id,
                             const void* pointer) {
    std::vector<int64_t> stride(original_stride, original_stride + spec.ndim);
    std::vector<int> permutation;
    if (spec.channels_last) permutation.push_back(1);
    for (int d = spec.ndim - 1; d > 1; --d) permutation.push_back(d);
    if (!spec.channels_last) permutation.push_back(1);
    permutation.push_back(0);
    int64_t z = 1;
    for (int d : permutation) {
        if (dim[d] == 1) stride[d] = z;
        else z *= dim[d];
    }
    auto dtype = spec.dtype == 0 ? CUDNN_DATA_FLOAT :
                 spec.dtype == 1 ? CUDNN_DATA_HALF : CUDNN_DATA_BFLOAT16;
    return cudnn_frontend::TensorBuilder().setDim(spec.ndim, dim)
        .setStrides(spec.ndim, stride.data()).setId(id).setAlignment(alignment(pointer))
        .setDataType(dtype).setVirtual(false).build();
}
std::string cache_key(const H3ConvSpec& s, const void* input, const void* weight, const void* output) {
    std::ostringstream key;
    key << s.ndim << ',' << s.dtype << ',' << s.transposed << ',' << s.allow_tf32
        << ',' << s.deterministic << ',' << s.channels_last << ':';
    for (auto a : {s.input_dims, s.input_strides, s.weight_dims, s.weight_strides,
                   s.output_dims, s.output_strides})
        for (int d = 0; d < s.ndim; ++d) key << a[d] << ',';
    for (auto a : {s.padding, s.stride, s.dilation})
        for (int d = 0; d < s.ndim - 2; ++d) key << a[d] << ',';
    key << int(alignment(input)) << ',' << int(alignment(weight)) << ',' << int(alignment(output));
    return key.str();
}
void execute(Context& ctx, const H3ConvSpec& spec, const void* input,
             const void* weight, void* output, const cudnn_frontend::ExecutionPlan& plan) {
    ctx.reserve(plan.getWorkspaceSize());
    void* pointers[] = {spec.transposed ? output : const_cast<void*>(input),
                       spec.transposed ? const_cast<void*>(input) : output,
                       const_cast<void*>(weight)};
    int64_t ids[] = {'x', 'y', 'w'};
    auto pack = cudnn_frontend::VariantPackBuilder()
        .setWorkspacePointer(plan.getWorkspaceSize() ? ctx.workspace : nullptr)
        .setDataPointers(3, pointers).setUids(3, ids).build();
    check(cudnnBackendExecute(ctx.handle, plan.get_raw_desc(), pack.get_raw_desc()));
    ctx.last_tag = plan.getTag();
}
bool reject(cudnnBackendDescriptor_t config, const H3ConvSpec& spec) {
    if (spec.deterministic && cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(config))
        return true;
    if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(config)) return true;
    return spec.dtype == 0 && !spec.allow_tf32 &&
        cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(config);
}
bool errata(const H3ConvSpec& spec, Context& ctx, const std::string& tag) {
    // Same built-in 3D errata as pinned Torch; no custom environment errata.
    static auto rules = nlohmann::json::parse(R"({"version":1,"rules":[
      {"rule_id":"163539","operation":"ConvFwd","engine":23,"cudnn_version_start":90800,"cudnn_version_end":91500},
      {"rule_id":"ConvBwdData","operation":"ConvBwdData","engine":23,"cudnn_version_start":8000,"cudnn_version_end":-1}]})");
    return spec.ndim > 4 && cudnn_frontend::check_errata(rules, tag, ctx.handle, [] { return true; });
}
void forward(Context& ctx, const H3ConvSpec& spec,
             const void* input, const void* weight, void* output, bool execute_kernel=true) {
    if ((spec.ndim != 4 && spec.ndim != 5) || spec.dtype < 0 || spec.dtype > 2)
        throw std::runtime_error("unsupported conv descriptor rank/dtype");
    const auto key = cache_key(spec, input, weight, output);
    auto existing = ctx.plans.find(key);
    if (existing != ctx.plans.end()) {
        if(execute_kernel)execute(ctx, spec, input, weight, output, *existing->second);
        return;
    }
    auto x = tensor(spec, spec.transposed ? spec.output_dims : spec.input_dims,
                    spec.transposed ? spec.output_strides : spec.input_strides,
                    'x', spec.transposed ? output : input);
    auto y = tensor(spec, spec.transposed ? spec.input_dims : spec.output_dims,
                    spec.transposed ? spec.input_strides : spec.output_strides,
                    'y', spec.transposed ? input : output);
    auto w = tensor(spec, spec.weight_dims, spec.weight_strides, 'w', weight);
    auto convolution = cudnn_frontend::ConvDescBuilder()
        .setDataType(CUDNN_DATA_FLOAT).setMathMode(CUDNN_CROSS_CORRELATION)
        .setNDims(spec.ndim - 2).setStrides(spec.ndim - 2, spec.stride)
        .setPrePadding(spec.ndim - 2, spec.padding).setPostPadding(spec.ndim - 2, spec.padding)
        .setDilation(spec.ndim - 2, spec.dilation).build();
    auto kind = spec.transposed ? CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
                               : CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
    auto operation = cudnn_frontend::OperationBuilder(kind)
        .setxDesc(x).setyDesc(y).setwDesc(w).setcDesc(convolution).build();
    std::array<cudnn_frontend::Operation const*, 1> operations = {&operation};
    auto graph = cudnn_frontend::OperationGraphBuilder().setHandle(ctx.handle)
        .setOperationGraph(operations.size(), operations.data()).build();
    std::string failures;
    for (bool fallback : {false, true}) {
        cudnn_frontend::EngineConfigList configs;
        if (!fallback) {
            auto heuristic = cudnn_frontend::EngineHeuristicsBuilder().setOperationGraph(graph)
                .setHeurMode(CUDNN_HEUR_MODE_INSTANT).build();
            configs = heuristic.getEngineConfig(heuristic.getEngineConfigCount());
        } else {
            auto heuristic = cudnn_frontend::EngineFallbackListBuilder().setOperationGraph(graph)
                .setOperation(kind).build();
            configs = heuristic.getFallbackList();
        }
        for (auto& config : configs) {
            if (reject(config->get_backend_descriptor(), spec)) continue;
            try {
                auto plan = std::make_unique<cudnn_frontend::ExecutionPlan>(
                    cudnn_frontend::ExecutionPlanBuilder().setHandle(ctx.handle)
                        .setEngineConfig(config, graph.getTag()).build());
                if (errata(spec, ctx, plan->getTag())) continue;
                if(execute_kernel)execute(ctx, spec, input, weight, output, *plan);
                ctx.plans.emplace(key, std::move(plan));
                return;
            } catch (const std::exception& exc) {
                failures += std::string(exc.what()) + "\n";
                cudaGetLastError();
            }
        }
    }
    throw std::runtime_error("cuDNN has no executable original-heuristic plan: " + failures);
}
}
extern "C" int h3_conv_create(void** pointer) {
    try { *pointer = new Context(); error.clear(); return 0; }
    catch (const std::exception& exc) { error = exc.what(); return -1; }
}
extern "C" int h3_conv_forward(void* context, const H3ConvSpec* spec,
    const void* input, const void* weight, void* output, void* stream) {
    try {
        auto& ctx = *static_cast<Context*>(context);
        check(cudnnSetStream(ctx.handle, static_cast<cudaStream_t>(stream)));
        forward(ctx, *spec, input, weight, output); error.clear(); return 0;
    } catch (const std::exception& exc) { error = exc.what(); return -1; }
}
extern "C" const char* h3_conv_error() { return error.c_str(); }
extern "C" const char* h3_conv_plan_tag(void* context) { return static_cast<Context*>(context)->last_tag.c_str(); }
extern "C" uint64_t h3_conv_workspace_bytes(void* context) { return static_cast<Context*>(context)->workspace_bytes; }
extern "C" int h3_conv_destroy(void* context) {
    try { delete static_cast<Context*>(context); return 0; }
    catch (const std::exception& exc) { error = exc.what(); return -1; }
}


extern "C" size_t q3_conv_cache_size(void* context){return static_cast<Context*>(context)->plans.size();}
extern "C" int q3_conv_cache_clear(void* context){
    try{auto& c=*static_cast<Context*>(context);c.plans.clear();return 0;}
    catch(const std::exception& e){error=e.what();return -1;}
}

// Build the original patch-embedding plan and register its library kernels
// in a discarded capture. No graph is instantiated or launched, no tensor is
// read, and no model parameters are needed. The worker owns an independent
// context until the serial conditioner adopts it before its first request.
extern "C" int q3_conv_prepare_isolated(int device, int64_t rows, void** out) {
    void* dummy = nullptr; cudaStream_t stream = nullptr; cudaGraph_t graph = nullptr;
    std::unique_ptr<Context> ctx;
    try {
        if (!out || rows <= 0 || rows > 524288) throw std::runtime_error("invalid patch preparation rows");
        cuda_check(cudaSetDevice(device));
        ctx = std::make_unique<Context>();
        cuda_check(cudaMalloc(&dummy,1u << 20));
        cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check(cudnnSetStream(ctx->handle,stream));
        H3ConvSpec spec{};
        spec.ndim=5; spec.dtype=2; spec.allow_tf32=1;
        const int64_t input_dims[]={rows,3,2,16,16}, weight_dims[]={1152,3,2,16,16}, output_dims[]={rows,1152,1,1,1};
        const int64_t stride[]={1536,512,256,16,1}, output_stride[]={1152,1,1,1,1};
        for(int i=0;i<5;++i){spec.input_dims[i]=input_dims[i];spec.weight_dims[i]=weight_dims[i];spec.output_dims[i]=output_dims[i];
            spec.input_strides[i]=stride[i];spec.weight_strides[i]=stride[i];spec.output_strides[i]=output_stride[i];}
        spec.stride[0]=2;spec.stride[1]=16;spec.stride[2]=16;
        for(int i=0;i<3;++i)spec.dilation[i]=1;
        const auto input=dummy;const auto weight=static_cast<char*>(dummy)+256;const auto output=static_cast<char*>(dummy)+512;
        forward(*ctx,spec,input,weight,output,false);
        size_t workspace=0;for(const auto& entry:ctx->plans)workspace=std::max(workspace,size_t(entry.second->getWorkspaceSize()));
        if(workspace>(2ull<<30))throw std::runtime_error("patch metadata workspace exceeds preparation budget");
        ctx->reserve(workspace);
        cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        forward(*ctx,spec,input,weight,output);
        cuda_check(cudaStreamEndCapture(stream,&graph));
        cuda_check(cudaGraphDestroy(graph));graph=nullptr;
        cuda_check(cudaStreamDestroy(stream));stream=nullptr;
        cuda_check(cudaFree(dummy));dummy=nullptr;
        *out=ctx.release();error.clear();return 0;
    } catch(const std::exception& e) {
        if(stream){cudaStreamCaptureStatus status{};cudaStreamIsCapturing(stream,&status);
            if(status!=cudaStreamCaptureStatusNone)cudaStreamEndCapture(stream,&graph);}
        if(graph)cudaGraphDestroy(graph);
        if(stream)cudaStreamDestroy(stream);
        if(dummy)cudaFree(dummy);
        error=e.what();return -1;
    }
}
extern "C" int q3_conv_adopt_isolated(void* destination, void* pointer) {
    try {
        if(!destination || !pointer)throw std::runtime_error("missing prepared convolution");
        auto& target=*static_cast<Context*>(destination);
        std::unique_ptr<Context> prepared(static_cast<Context*>(pointer));
        if(!target.plans.empty())throw std::runtime_error("convolution adoption requires an unused owner");
        std::swap(target.handle,prepared->handle);target.plans.swap(prepared->plans);
        std::swap(target.workspace,prepared->workspace);std::swap(target.workspace_bytes,prepared->workspace_bytes);
        target.last_tag.swap(prepared->last_tag);error.clear();return 0;
    }catch(const std::exception& e){error=e.what();return -1;}
}
