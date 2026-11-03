// Adapted from PyTorch cf30153c4c131c8164ee7798e5022d810682e2cb
// aten/src/ATen/native/cudnn/Conv_v8.cpp and ATen/cudnn/Descriptors.h.
// See licenses/PYTORCH-LICENSE and licenses/NOTICE.md.
#include "conv.h"
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_get_plan.h>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
struct WorkspaceLimit : std::runtime_error {using std::runtime_error::runtime_error;};
struct WorkspaceOutOfMemory : std::runtime_error {using std::runtime_error::runtime_error;};
infra_conv_oom_handler oom_handler = nullptr;
struct Context {
    cudnnHandle_t handle{};
    std::unordered_map<std::string, std::unique_ptr<cudnn_frontend::ExecutionPlan>> plans;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    std::string last_tag;
    const size_t plan_limit;
    const size_t workspace_limit;
    std::unordered_map<std::string,uint64_t> ages;
    uint64_t clock=0,hits=0,misses=0,evictions=0,workspace_allocations=0;
    Context(size_t plans_limit,size_t bytes_limit) : plan_limit(plans_limit), workspace_limit(bytes_limit) { check(cudnnCreate(&handle)); }
    ~Context() {
        plans.clear();
        if (workspace) cudaFree(workspace);
        if (handle) cudnnDestroy(handle);
    }
    void reserve(size_t count) {
        if (count <= workspace_bytes) return;
        if (count > workspace_limit) throw WorkspaceLimit("cuDNN workspace exceeds caller budget");
        void* next = nullptr;
        cudaError_t status = cudaMalloc(&next, count);
        if (status != cudaSuccess && oom_handler) {
            cudaGetLastError();
            oom_handler();
            status = cudaMalloc(&next, count);
            if (status != cudaSuccess) {
                cudaGetLastError();
                throw WorkspaceOutOfMemory("cuDNN workspace allocation failed after releasing cached memory");
            }
        }
        cuda_check(status);
        if (workspace) cuda_check(cudaFree(workspace));
        workspace = next;
        workspace_bytes = count; ++workspace_allocations;
    }
};
uint8_t alignment(const void* pointer) {
    uint8_t result = 1;
    for (; result < 32; result *= 2)
        if (reinterpret_cast<uintptr_t>(pointer) % (result * 2)) return result;
    return result;
}
cudnn_frontend::Tensor tensor(const InfraConvSpec& spec, const int64_t* dim,
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
std::string cache_key(const InfraConvSpec& s, const void* input, const void* weight, const void* output) {
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
void execute(Context& ctx, const InfraConvSpec& spec, const void* input,
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
bool reject(cudnnBackendDescriptor_t config, const InfraConvSpec& spec) {
    if (spec.deterministic && cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(config))
        return true;
    if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(config)) return true;
    return spec.dtype == 0 && !spec.allow_tf32 &&
        cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(config);
}
bool errata(const InfraConvSpec& spec, Context& ctx, const std::string& tag) {
    // Same built-in 3D errata as pinned Torch; no custom environment errata.
    static auto rules = nlohmann::json::parse(R"({"version":1,"rules":[
      {"rule_id":"163539","operation":"ConvFwd","engine":23,"cudnn_version_start":90800,"cudnn_version_end":91500},
      {"rule_id":"ConvBwdData","operation":"ConvBwdData","engine":23,"cudnn_version_start":8000,"cudnn_version_end":-1}]})");
    return spec.ndim > 4 && cudnn_frontend::check_errata(rules, tag, ctx.handle, [] { return true; });
}
void forward(Context& ctx, const InfraConvSpec& spec,
             const void* input, const void* weight, void* output) {
    if ((spec.ndim != 4 && spec.ndim != 5) || spec.dtype < 0 || spec.dtype > 2)
        throw std::runtime_error("unsupported conv descriptor rank/dtype");
    const auto key = cache_key(spec, input, weight, output);
    auto existing = ctx.plans.find(key);
    if (existing != ctx.plans.end()) {
        ++ctx.hits;ctx.ages[key]=++ctx.clock;
        execute(ctx, spec, input, weight, output, *existing->second);
        return;
    }
    ++ctx.misses;
    // NATIVE_CONV_PLAN_TRACE=1 reports each first-use plan build on stderr (startup diagnostics).
    static const bool trace = std::getenv("NATIVE_CONV_PLAN_TRACE") != nullptr;
    const auto build_started = std::chrono::steady_clock::now();
    struct Report {
        bool on; std::chrono::steady_clock::time_point started; int ndim;
        ~Report() { if (on) std::fprintf(stderr, "native_conv_plan status=built ndim=%d seconds=%.6f\n", ndim,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()); }
    } report{trace, build_started, spec.ndim};
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
                execute(ctx, spec, input, weight, output, *plan);
                if (ctx.plans.size()>=ctx.plan_limit) {
                    auto oldest=ctx.ages.begin();for(auto it=ctx.ages.begin();it!=ctx.ages.end();++it)if(it->second<oldest->second)oldest=it;
                    cudaStream_t stream;check(cudnnGetStream(ctx.handle,&stream));cuda_check(cudaStreamSynchronize(stream));
                    ctx.plans.erase(oldest->first);ctx.ages.erase(oldest);++ctx.evictions;
                }
                ctx.ages[key]=++ctx.clock;ctx.plans.emplace(key, std::move(plan));
                return;
            } catch (const WorkspaceLimit&) { throw;
            } catch (const WorkspaceOutOfMemory&) { throw;
            } catch (const std::exception& exc) {
                failures += std::string(exc.what()) + "\n";
                cudaGetLastError();
            }
        }
    }
    throw std::runtime_error("cuDNN has no executable original-heuristic plan: " + failures);
}
}
extern "C" void infra_conv_set_oom_handler(infra_conv_oom_handler handler) { oom_handler = handler; }
extern "C" int infra_conv_create(void** pointer, size_t plan_limit, size_t workspace_limit) {
    if (!pointer || !plan_limit || !workspace_limit) return -1;
    try { *pointer = new Context(plan_limit,workspace_limit); error.clear(); return 0; }
    catch (const std::exception& exc) { error = exc.what(); return -1; }
}
extern "C" int infra_conv_forward(void* context, const InfraConvSpec* spec,
    const void* input, const void* weight, void* output, void* stream) {
    try {
        auto& ctx = *static_cast<Context*>(context);
        check(cudnnSetStream(ctx.handle, static_cast<cudaStream_t>(stream)));
        forward(ctx, *spec, input, weight, output); error.clear(); return 0;
    } catch (const std::exception& exc) { error = exc.what(); return -1; }
}
extern "C" const char* infra_conv_error() { return error.c_str(); }
extern "C" const char* infra_conv_plan_tag(void* context) { return static_cast<Context*>(context)->last_tag.c_str(); }
extern "C" uint64_t infra_conv_workspace_bytes(void* context) { return static_cast<Context*>(context)->workspace_bytes; }
extern "C" int infra_conv_release_workspace(void* context, void* stream) {
    try {
        auto& c=*static_cast<Context*>(context);
        cuda_check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
        if (c.workspace) cuda_check(cudaFree(c.workspace));
        c.workspace=nullptr;c.workspace_bytes=0;
        error.clear();return 0;
    } catch (const std::exception& exc) { error=exc.what();return -1; }
}
extern "C" int infra_conv_destroy(void* context) {
    try { delete static_cast<Context*>(context); return 0; }
    catch (const std::exception& exc) { error = exc.what(); return -1; }
}


extern "C" int infra_conv_cache_stats(void* context,uint64_t* out) {
    const auto& c=*static_cast<Context*>(context);
    uint64_t values[]={c.plans.size(),c.plan_limit,c.hits,c.misses,c.evictions,c.workspace_bytes,c.workspace_limit,c.workspace_allocations};
    for(size_t i=0;i<8;++i)out[i]=values[i];return 0;
}
