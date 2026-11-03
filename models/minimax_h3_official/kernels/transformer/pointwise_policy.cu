// Host-only request-scoped switch; no kernel mathematics.
#include <cstdint>
namespace {
thread_local int enabled = 1;
thread_local uint64_t baseline_calls = 0, optimized_calls = 0;
}
extern "C" void h3_pointwise_opt_set_mode(int mode) {
    enabled = mode; baseline_calls = 0; optimized_calls = 0;
}
extern "C" int h3_pointwise_opt_enabled() {
    if (enabled) ++optimized_calls; else ++baseline_calls;
    return enabled;
}
extern "C" uint64_t h3_pointwise_opt_calls(int mode) {
    return mode ? optimized_calls : baseline_calls;
}
