// Adapted from PyTorch cf30153c4c131c8164ee7798e5022d810682e2cb,
// ATen/native/cpu/DistributionTemplates.h::NormalFill16<float,true>.
// See ../NOTICE.md and ../reference_sources/PYTORCH-LICENSE.
// The original standalone AVX polynomial header is a build-time dependency;
// this object has no ATen/PyTorch runtime dependency.
#include <ATen/native/cpu/avx_mathfun.h>
#include <cstddef>

extern "C" void infra_normal_transform16(float* data, size_t count) {
    const auto two_pi = _mm256_set1_ps(float(2.0 * 3.14159265358979323846));
    const auto one = _mm256_set1_ps(1.0f);
    const auto minus_two = _mm256_set1_ps(-2.0f);
    const auto zero = _mm256_set1_ps(0.0f);
    for (size_t i = 0; i < count; i += 16) {
        auto* p = data + i;
        const auto u1 = _mm256_sub_ps(one, _mm256_loadu_ps(p));
        const auto u2 = _mm256_loadu_ps(p + 8);
        const auto radius = _mm256_sqrt_ps(_mm256_mul_ps(minus_two, log256_ps(u1)));
        const auto theta = _mm256_mul_ps(two_pi, u2);
        __m256 sine, cosine;
        sincos256_ps(theta, &sine, &cosine);
        _mm256_storeu_ps(p, _mm256_fmadd_ps(_mm256_mul_ps(radius, cosine), one, zero));
        _mm256_storeu_ps(p + 8, _mm256_fmadd_ps(_mm256_mul_ps(radius, sine), one, zero));
    }
}

