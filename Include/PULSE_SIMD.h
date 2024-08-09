#ifndef __PULSE_SIMD
#define __PULSE_SIMD

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>

#define __PULSE_MIN(x, y) ((x) < (y) ? (x) : (y))
#define __PULSE_MAX(x, y) ((x) > (y) ? (x) : (y))

#if defined(__AVX512F__)
#define __PULSE_SIMD_AVX512
#define __PULSE_SIMD_SUPPORTED
#elif defined(__AVX__) || defined(__AVX2__)
#define __PULSE_SIMD_AVX
#define __PULSE_SIMD_SUPPORTED
#elif defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)
#define __PULSE_SIMD_SSE
#define __PULSE_SIMD_SUPPORTED
#else
#define __PULSE_SIMD_NONE
#endif

#if defined(__i386__) || defined(__x86_64__)
#define __PULSE_ARCH 0
#include <immintrin.h>
static inline __m128 __PULSE_SIMD_X86_REDUCE_ADD_128(__m128 x) { __m128 odd = _mm_movehdup_ps(x); __m128 sum = _mm_add_ps(x, odd); return _mm_add_ss(_mm_movehl_ps(odd, sum), sum);};
static inline __m128 __PULSE_SIMD_X86_REDUCE_ADD_256(__m256 x) { __m128 tmp = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)); return __PULSE_SIMD_X86_REDUCE_ADD_128(tmp);};
static inline __m256 __PULSE_SIMD_X86_ZERO_R_256(__m256 x, int n) {return _mm256_blendv_ps(x, _mm256_setzero_ps(), _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set_epi32(7,6,5,4,3,2,1,0), _mm256_set1_epi32(7 - n))));};
static inline __m128 __PULSE_SIMD_X86_ZERO_R_128(__m128 x, int n) {return _mm_blendv_ps(x, _mm_setzero_ps(), _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_set_epi32(3,2,1,0), _mm_set1_epi32(3 - n))));};
#endif
#endif
