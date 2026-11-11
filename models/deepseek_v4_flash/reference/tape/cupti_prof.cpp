// Kernel activity profiler (CUPTI), injected with CUDA_INJECTION64_PATH: every kernel the process
// runs, including those replayed from CUDA graphs, as one JSON line (name, device start/end in ns,
// grid, block, graph id) in $PROF_DIR/prof-<pid>.jsonl. Records are flushed from CUPTI's buffer
// callback and at exit. Used to compare SGLang's and the native port's decode kernel by kernel.
#include <cupti.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unistd.h>

namespace {
FILE *g_out = nullptr;
std::mutex g_mutex;
constexpr size_t kBuffer = 8 << 20;

void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size, size_t *max_records) {
  *buffer = static_cast<uint8_t *>(aligned_alloc(8, kBuffer));
  *size = kBuffer;
  *max_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t *buffer, size_t, size_t valid) {
  CUpti_Activity *record = nullptr;
  std::lock_guard<std::mutex> lock(g_mutex);
  while (cuptiActivityGetNextRecord(buffer, valid, &record) == CUPTI_SUCCESS) {
    if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL && record->kind != CUPTI_ACTIVITY_KIND_KERNEL) continue;
    auto *k = reinterpret_cast<CUpti_ActivityKernel9 *>(record);
    if (g_out) {
      fprintf(g_out, "{\"name\":\"%s\",\"start\":%llu,\"end\":%llu,\"grid\":[%d,%d,%d],\"block\":[%d,%d,%d],\"graph\":%u,\"stream\":%u}\n",
              k->name ? k->name : "", (unsigned long long)k->start, (unsigned long long)k->end, k->gridX, k->gridY, k->gridZ,
              k->blockX, k->blockY, k->blockZ, k->graphId, k->streamId);
    }
  }
  if (g_out) fflush(g_out);
  free(buffer);
}

void flush_at_exit() {
  cuptiActivityFlushAll(1);
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_out) fclose(g_out);
  g_out = nullptr;
}
}  // namespace

extern "C" __attribute__((visibility("default"))) int InitializeInjection() {
  const char *dir = getenv("PROF_DIR");
  std::string path = std::string(dir ? dir : "/tmp") + "/prof-" + std::to_string(getpid()) + ".jsonl";
  g_out = fopen(path.c_str(), "w");
  cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cuptiActivityFlushPeriod(500);  // servers are stopped by signal: flush records as they go
  atexit(flush_at_exit);
  return 1;
}

// Explicit flush for long-running processes (call through dlsym, or rely on buffer completion).
extern "C" __attribute__((visibility("default"))) void prof_flush() { cuptiActivityFlushAll(1); }
