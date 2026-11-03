// Preserve the model ABI; graph construction and persistent plan reuse are shared.
#define infra_sdpa_error wan_sdpa_error
#define infra_sdpa_prepare wan_sdpa_prepare
#define infra_sdpa_forward wan_sdpa_forward
#include "../../../core_infra/kernels/sdpa.cpp"
