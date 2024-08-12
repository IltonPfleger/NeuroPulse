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
#define __PULSE_SIMD_CHUNK_SIZE 512
#define __PULSE_SIMD_SUPPORTED
#elif defined(__AVX__) || defined(__AVX2__)
#define __PULSE_SIMD_CHUNK_SIZE 256
#define __PULSE_SIMD_SUPPORTED
#elif defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)
#define __PULSE_SIMD_CHUNK_SIZE 128
#define __PULSE_SIMD_SUPPORTED
#else
#define __PULSE_SIMD_CHUNK_SIZE 0
#endif
#define __PULSE_FLOAT_PS_SIZE 32
#define __PULSE_FLOAT_PD_SIZE 64

#define __PULSE_SIMD_N_PER_CHUNK __PULSE_SIMD_CHUNK_SIZE/__PULSE_FLOAT_PS_SIZE
#if defined(__PULSE_SIMD_SUPPORTED)
#define __PULSE_SIMD_CHECK(x) x
#else
#define __PULSE_SIMD_CHECK(x) (printf("ERROR: PULSE SIMD It's Not Supported On This Device"), exit(1))
#endif



#if defined(__i386__) || defined(__x86_64__)
#define __PULSE_ARCH 0
#include <immintrin.h>

static inline __m256 __PULSE_SIMD_X86_ZERO_R_256(__m256 x, int n) {return _mm256_blendv_ps(x, _mm256_setzero_ps(), _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set_epi32(7,6,5,4,3,2,1,0), _mm256_set1_epi32(7 - n))));};
static inline __m128 __PULSE_SIMD_X86_ZERO_R_128(__m128 x, int n) {return _mm_blendv_ps(x, _mm_setzero_ps(), _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_set_epi32(3,2,1,0), _mm_set1_epi32(3 - n))));};
static inline __m128 __PULSE_SIMD_X86_REDUCE_ADD_128(__m128 x) { __m128 odd = _mm_movehdup_ps(x); __m128 sum = _mm_add_ps(x, odd); return _mm_add_ss(_mm_movehl_ps(odd, sum), sum);};
static inline __m128 __PULSE_SIMD_X86_REDUCE_ADD_256(__m256 x) { __m128 tmp = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)); return __PULSE_SIMD_X86_REDUCE_ADD_128(tmp);};

#define __PULSE_SIMD_REDUCE_ADD(X) _Generic((X), \
__m128: __PULSE_SIMD_X86_REDUCE_ADD_128, \
__m256: __PULSE_SIMD_X86_REDUCE_ADD_256)(X)

#define __PULSE_SIMD_ZERO_R(X, Y) _Generic((X), \
		__m128: __PULSE_SIMD_X86_ZERO_R_128, \
		__m256: __PULSE_SIMD_X86_ZERO_R_256)(X, Y)

#define __PULSE_SIMD_MUL(X, Y) _Generic((X), \
		__m128: _mm_mul_ps, \
		__m256: _mm256_mul_ps)(X, Y)

#define __PULSE_SIMD_ADD(X, Y) _Generic((X), \
		__m128: _mm_add_ps, \
		__m256: _mm256_add_ps)(X, Y)

#define __PULSE_SIMD_STORE(X, Y) _Generic((Y), \
		__m128: _mm_storeu_ps, \
		__m256: _mm256_storeu_ps)(X, Y)

#define __PULSE_SIMD_MADD(X, Y, Z) _Generic((X), \
		__m128: _mm_fmadd_ps, \
		__m256: _mm256_fmadd_ps)(X, Y, Z)

#define __PULSE_SIMD_TO_FLOAT(x) _mm_cvtss_f32(x)

#if __PULSE_ARCH == 0 && __PULSE_SIMD_CHUNK_SIZE == 256
#define __PULSE_SIMD_DATATYPE __m256
#define __PULSE_SIMD_LOAD(x) _mm256_loadu_ps(x)
#define __PULSE_SIMD_ALLIGNED_LOAD(x) _mm256_load_ps(x)
#define __PULSE_SIMD_ZERO() _mm256_setzero_ps()
#define __PULSE_SIMD_SET_ALL(x) _mm256_set1_ps(x)
#endif

#if __PULSE_ARCH == 0 && __PULSE_SIMD_CHUNK_SIZE == 128
#define __PULSE_SIMD_DATATYPE __128
#define __PULSE_SIMD_LOAD(x) _mm_loadu_ps(x)
#define __PULSE_SIMD_ALLIGNED_LOAD(x) _mm_load_ps(x)
#define __PULSE_SIMD_ZERO() _mm_setzero_ps()
#define __PULSE_SIMD_SET_ALL(x) _mm_set1_ps(x)
#endif
#endif
#endif
