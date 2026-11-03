#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace h3_comm_contract {
inline void require(bool ok, const char *message) {
    if (!ok) throw std::invalid_argument(message);
}
inline size_t add(size_t a, size_t b) {
    require(b <= std::numeric_limits<size_t>::max() - a, "element addition overflow");
    return a + b;
}
inline size_t mul(size_t a, size_t b) {
    require(a == 0 || b <= std::numeric_limits<size_t>::max() / a, "byte/count multiplication overflow");
    return a * b;
}
inline void rank_world(int rank, int world) {
    require(world == 1 || world == 2 || world == 4 || world == 8, "world must be 1, 2, 4 or 8");
    require(rank >= 0 && rank < world, "rank outside world");
}
inline size_t dtype_bytes(int dtype) {
    switch (dtype) {
        case 0: return 1;
        case 1: case 2: return 2;
        case 3: case 5: return 4;
        case 4: return 8;
        default: throw std::invalid_argument("unknown dtype");
    }
}
inline std::vector<size_t> offsets(const std::vector<size_t>& counts) {
    std::vector<size_t> out(counts.size());
    size_t total = 0;
    for (size_t i = 0; i < counts.size(); ++i) { out[i] = total; total = add(total, counts[i]); }
    return out;
}
inline void intervals(const size_t *counts, const size_t *offsets, int world,
                      size_t capacity, size_t element_bytes, bool disjoint) {
    require(counts && offsets, "null count/offset array");
    mul(capacity, element_bytes);
    std::vector<std::pair<size_t, size_t>> spans;
    for (int i = 0; i < world; ++i) {
        const auto end = add(offsets[i], counts[i]);
        require(end <= capacity, "peer interval exceeds buffer capacity");
        if (counts[i]) spans.emplace_back(offsets[i], end);
    }
    if (disjoint) {
        std::sort(spans.begin(), spans.end());
        for (size_t i = 1; i < spans.size(); ++i)
            require(spans[i - 1].second <= spans[i].first, "receive intervals overlap");
    }
}
inline void pointer(const void *p, size_t bytes) {
    require(bytes == 0 || p != nullptr, "nonempty buffer is null");
    require(bytes <= std::numeric_limits<uintptr_t>::max() - reinterpret_cast<uintptr_t>(p),
            "device address overflow");
}
inline void disjoint(const void *a, size_t a_bytes, const void *b, size_t b_bytes,
                     bool allow_exact = false) {
    pointer(a, a_bytes); pointer(b, b_bytes);
    if (!a_bytes || !b_bytes || (allow_exact && a == b && a_bytes == b_bytes)) return;
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    require(x + a_bytes <= y || y + b_bytes <= x, "send/receive storage overlaps");
}
// Fixed-size metadata allows debug checks to detect operation/count disagreement
// before entering differently sized point-to-point exchanges.
constexpr size_t max_world = 8;
constexpr size_t meta_header_words = 6;
constexpr size_t meta_send_offset = meta_header_words;
constexpr size_t meta_recv_offset = meta_send_offset + max_world;
constexpr size_t meta_rows_offset = meta_recv_offset + max_world;
constexpr size_t meta_words = meta_rows_offset + max_world;
using Metadata = std::vector<uint64_t>;
inline void agreement(const Metadata& all, int world) {
    rank_world(0, world);
    require(all.size() == static_cast<size_t>(world) * meta_words, "metadata extent mismatch");
    for (int r = 0; r < world; ++r) {
        const auto *m = all.data() + r * meta_words;
        for (size_t i = 0; i < meta_header_words; ++i)
            require(m[i] == all[i], "collective operation/sequence/type/shape mismatch");
    }
    const auto op = all[1];
    if (op == 1) {
        for (int src = 0; src < world; ++src) for (int dst = 0; dst < world; ++dst)
            require(all[src * meta_words + meta_send_offset + dst] == all[dst * meta_words + meta_recv_offset + src],
                    "alltoallv peer send/receive count mismatch");
    } else if (op == 2) {
        for (int r = 0; r < world; ++r) for (int p = 0; p < world; ++p)
            require(all[r * meta_words + meta_rows_offset + p] == all[meta_rows_offset + p], "allgatherv row counts mismatch");
    }
}
} // namespace h3_comm_contract
