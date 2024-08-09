#ifndef __PULSE_SIMD
#define __PULSE_SIMD

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>

#define __PULSE_MIN(x, y) ((x) < (y) ? (x) : (y))
#define __PULSE_MAX(x, y) ((x) > (y) ? (x) : (y))
//#define __PULSE_CACHE_LINE_ALIGNED_ALLOC(size) aligned_alloc(PULSE_CFLAGS_CacheLineSize, size%PULSE_CFLAGS_CacheLineSize == 0 ? size : size + size%PULSE_CFLAGS_CacheLineSize)

#if defined(__AVX512F__)
#define __PULSE_SIMD_CHUNK_SIZE 512
#elif defined(__AVX__) || defined(__AVX2__)
#define __PULSE_SIMD_CHUNK_SIZE 256
#elif defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)
#define __PULSE_SIMD_CHUNK_SIZE 128
#else
#define __PULSE_SIMD_CHUNK_SIZE 0
#endif
#define __PULSE_FLOAT_PS_SIZE 32
#define __PULSE_FLOAT_PD_SIZE 64

#define __PULSE_SIMD_N_PER_CHUNK __PULSE_SIMD_CHUNK_SIZE/__PULSE_FLOAT_PS_SIZE
#define __PULSE_SIMD_N_PER_CACHE_LINE (PULSE_CFLAGS_CacheLineSize/(__PULSE_SIMD_CHUNK_SIZE/8))*__PULSE_SIMD_N_PER_CHUNK




#if defined(__i386__) || defined(__x86_64__)
#define __PULSE_ARCH 0
#include <immintrin.h>
static inline __m128 __PULSE_SIMD_X86_REDUCE_ADD_128(__m128 x) { __m128 odd = _mm_movehdup_ps(x); __m128 sum = _mm_add_ps(x, odd); return _mm_add_ss(_mm_movehl_ps(odd, sum), sum);};
static inline __m128 __PULSE_SIMD_X86_REDUCE_ADD_256(__m256 x) { __m128 tmp = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)); return __PULSE_SIMD_X86_REDUCE_ADD_128(tmp);};
static inline __m256 __PULSE_SIMD_X86_ZERO_R_256(__m256 x, int n) {return _mm256_blendv_ps(x, _mm256_setzero_ps(), _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set_epi32(7,6,5,4,3,2,1,0), _mm256_set1_epi32(7 - n))));};
static inline __m128 __PULSE_SIMD_X86_ZERO_R_128(__m128 x, int n) {return _mm_blendv_ps(x, _mm_setzero_ps(), _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_set_epi32(3,2,1,0), _mm_set1_epi32(3 - n))));};
#endif


#if __PULSE_ARCH == 0 && __PULSE_SIMD_CHUNK_SIZE == 256
#define __PULSE_SIMD_BIGEST_DATA_TYPE __m256
#define __PULSE_SIMD_LOWEST_DATA_TYPE __m128
#define __PULSE_SIMD_LOAD(x) _mm256_loadu_ps(x)
#define __PULSE_SIMD_MUL(x, y) _mm256_mul_ps(x, y)
#define __PULSE_SIMD_ADD(x, y) _mm256_add_ps(x, y)
#define __PULSE_SIMD_ZERO() _mm256_setzero_ps()
#define __PULSE_SIMD_SET_ALL(x) _mm256_set1_ps(x)
#define __PULSE_SIMD_GET(dst, x) _mm256_storeu_ps(dst, x)
#define __PULSE_SIMD_REDUCE_ADD(x) __PULSE_SIMD_X86_REDUCE_ADD_256(x)
#define __PULSE_SIMD_TO_FLOAT(x) _mm_cvtss_f32(x)
#define __PULSE_SIMD_ZERO_R(x, n) __PULSE_SIMD_X86_ZERO_R_256(x, n)
#define __PULSE_SIMD_MADD(x, y, z) _mm256_fmadd_ps(x, y, z)
#define __PULSE_SIMD_PREFETCH(x) _mm_prefetch(x, _MM_HINT_T0)
#endif

#if __PULSE_ARCH == 0 && __PULSE_SIMD_CHUNK_SIZE == 128
#define __PULSE_SIMD_BIGEST_DATA_TYPE __m128
#define __PULSE_SIMD_LOWEST_DATA_TYPE __m128
#define __PULSE_SIMD_LOAD(x) _mm_loadu_ps(x)
#define __PULSE_SIMD_MUL(x, y) _mm_mul_ps(x, y)
#define __PULSE_SIMD_ADD(x, y) _mm_add_ps(x, y)
#define __PULSE_SIMD_ZERO() _mm_setzero_ps()
#define __PULSE_SIMD_SET_ALL(x) _mm_set1_ps(x)
#define __PULSE_SIMD_GET(dst, x) _mm_storeu_ps(dst, x)
#define __PULSE_SIMD_REDUCE_ADD(x) __PULSE_SIMD_X86_REDUCE_ADD_128(x)
#define __PULSE_SIMD_TO_FLOAT(x) _mm_cvtss_f32(x)
#define __PULSE_SIMD_ZERO_R(x, n) __PULSE_SIMD_X86_ZERO_R_128(x, n)
#endif

//static void __PULSE_ADD_MxTM(float * m1, float * m2, float * dst, int m1R, int m2R, int K)
//{
//	__PULSE_SIMD_BIGEST_DATA_TYPE localM1[__PULSE_SIMD_N_PER_CHUNK];
//	__PULSE_SIMD_BIGEST_DATA_TYPE localM2[__PULSE_SIMD_N_PER_CHUNK];
//	__PULSE_SIMD_BIGEST_DATA_TYPE simd_mul;
//	__PULSE_SIMD_LOWEST_DATA_TYPE simd_sum;
//	alignas(PULSE_CFLAGS_CacheLineSize) float localDst[__PULSE_SIMD_N_PER_CHUNK][__PULSE_SIMD_N_PER_CHUNK];
//	const __PULSE_SIMD_BIGEST_DATA_TYPE zero = __PULSE_SIMD_ZERO();
//	const int R1 = m1R%__PULSE_SIMD_N_PER_CHUNK == 0 ? m1R : m1R + __PULSE_SIMD_N_PER_CHUNK;
//	const int R2 = m2R%__PULSE_SIMD_N_PER_CHUNK == 0 ? m2R : m2R + __PULSE_SIMD_N_PER_CHUNK;
//
//#pragma omp parallel for schedule(static) private(localM1, localM2, localDst, simd_mul, simd_sum)
//	for(int i = 0; i < R1; i += __PULSE_SIMD_N_PER_CHUNK)
//	{
//		const int TR1 = __PULSE_MIN(m1R - i, __PULSE_SIMD_N_PER_CHUNK);
//		for(int j = 0; j < R2; j += __PULSE_SIMD_N_PER_CHUNK)
//		{
//			const int TR2 = __PULSE_MIN(m2R - j, __PULSE_SIMD_N_PER_CHUNK);
//			for(int k = 0; k < K; k += __PULSE_SIMD_N_PER_CHUNK)
//			{
//				for(int l = 0; l < __PULSE_SIMD_N_PER_CHUNK; l++)
//				{
//					localM1[l] = __PULSE_SIMD_LOAD(m1 + (i + l)*K + k);
//					localM2[l] = __PULSE_SIMD_LOAD(m2 + (j + l)*K + k);
//
//					for(int m = 0; m < __PULSE_SIMD_N_PER_CHUNK; m++)
//						localDst[l][m] = 0.f;
//					if(k + __PULSE_SIMD_N_PER_CHUNK > K)
//					{
//						localM1[l] = __PULSE_SIMD_ZERO_R(localM1[l], K - k);
//						localM2[l] = __PULSE_SIMD_ZERO_R(localM2[l], K - k);
//					}
//				}
//				for(int l = 0; l < TR1; l++)
//				{
//					for(int m = 0; m < TR2; m++)
//					{
//						simd_mul = __PULSE_SIMD_MUL(localM1[l], localM2[m]);
//						simd_sum = __PULSE_SIMD_REDUCE_ADD(simd_mul);
//						localDst[l][m] += __PULSE_SIMD_TO_FLOAT(simd_sum);
//					}
//				}
//				for(int l = 0; l < TR1; l++)
//				{
//					for(int m = 0; m < TR2; m++)
//					{
//						dst[(i + l) * m2R + (j + m)] += localDst[l][m];
//					}
//				}
//			}
//		}
//	}
//}
//
//static void __PULSE_MxTM(float * m1, float * m2, float * dst, int m1R, int m2R, int K)
//{
//	memset(dst, 0, sizeof(float)*m1R*m2R);
//	__PULSE_ADD_MxTM(m1,  m2, dst,m1R, m2R,K);
//}


static void __PULSE_SIMD_1MNxMMN(float * m1, float * m2, float * dst, int m, int n)
{
	__PULSE_SIMD_BIGEST_DATA_TYPE m1Dst, m2Dst, sum;
	for(int i = 0; i < m; i++)
	{
		const int wi = i*n;
		sum = __PULSE_SIMD_ZERO();
		for(int j = 0; j < n; j += __PULSE_SIMD_N_PER_CHUNK)
		{
			m2Dst = __PULSE_SIMD_LOAD(m2 + wi + j);
			m1Dst = __PULSE_SIMD_LOAD(m1 + j);
			if(j + __PULSE_SIMD_N_PER_CHUNK > n)
			{
				m1Dst = __PULSE_SIMD_ZERO_R(m1Dst, __PULSE_SIMD_N_PER_CHUNK - n - j);
				m2Dst = __PULSE_SIMD_ZERO_R(m2Dst, __PULSE_SIMD_N_PER_CHUNK - n - j);
			}
			sum = __PULSE_SIMD_MADD(m1Dst, m2Dst ,sum);
		}
		dst[i] = __PULSE_SIMD_TO_FLOAT(__PULSE_SIMD_REDUCE_ADD(sum));
	}
}


static void __PULSE_SIMD_1MNxMMN_ADD(float * m1, float * m2, float * beta, float * dst, int m, int n)
{
	__PULSE_SIMD_BIGEST_DATA_TYPE m1Dst, m2Dst, sum;
	for(int i = 0; i < m; i++)
	{
		const int wi = i*n;
		sum = __PULSE_SIMD_ZERO();
		for(int j = 0; j < n; j += __PULSE_SIMD_N_PER_CHUNK)
		{
			m2Dst = __PULSE_SIMD_LOAD(m2 + wi + j);
			m1Dst = __PULSE_SIMD_LOAD(m1 + j);
			if(j + __PULSE_SIMD_N_PER_CHUNK > n)
			{
				m1Dst = __PULSE_SIMD_ZERO_R(m1Dst, __PULSE_SIMD_N_PER_CHUNK - n - j);
				m2Dst = __PULSE_SIMD_ZERO_R(m2Dst, __PULSE_SIMD_N_PER_CHUNK - n - j);
			}
			sum = __PULSE_SIMD_MADD(m1Dst, m2Dst ,sum);
		}
		dst[i] = __PULSE_SIMD_TO_FLOAT(__PULSE_SIMD_REDUCE_ADD(sum)) + beta[i];
	}
}

static void __PULSE_SIMD_ADD_1MNxMMN(float * m1, float * m2, float * dst, int m, int n)
{
	__PULSE_SIMD_BIGEST_DATA_TYPE m1Dst, m2Dst, sum;
	for(int i = 0; i < m; i++)
	{
		const int wi = i*n;
		sum = __PULSE_SIMD_ZERO();
		for(int j = 0; j < n; j += __PULSE_SIMD_N_PER_CHUNK)
		{
			m2Dst = __PULSE_SIMD_LOAD(m2 + wi + j);
			m1Dst = __PULSE_SIMD_LOAD(m1 + j);
			if(j + __PULSE_SIMD_N_PER_CHUNK > n)
			{
				m1Dst = __PULSE_SIMD_ZERO_R(m1Dst, __PULSE_SIMD_N_PER_CHUNK - n - j);
				m2Dst = __PULSE_SIMD_ZERO_R(m2Dst, __PULSE_SIMD_N_PER_CHUNK - n - j);
			}
			sum = __PULSE_SIMD_MADD(m1Dst, m2Dst ,sum);
		}
		dst[i] += __PULSE_SIMD_TO_FLOAT(__PULSE_SIMD_REDUCE_ADD(sum));
	}
}





//static void __PULSE_ADD_1MNxMMN(float * m1, float * m2, float * dst, int n, int m)
//{
//	__PULSE_SIMD_BIGEST_DATA_TYPE m1Dst, m2Dst, sum;
//#pragma omp parallel for schedule(static) private(m1Dst, m2Dst, sum)
//	for(int i = 0; i < m; i++)
//	{
//		const int wi = i*n;
//		sum = __PULSE_SIMD_ZERO();
//		for(int j = 0; j < n; j += __PULSE_SIMD_N_PER_CHUNK)
//		{
//			m2Dst = __PULSE_SIMD_LOAD(m2 + wi + j);
//			m1Dst = __PULSE_SIMD_LOAD(m1 + j);
//			if(j + __PULSE_SIMD_N_PER_CHUNK > n)
//			{
//				m1Dst = __PULSE_SIMD_N_TO_ZEROS(m1Dst, __PULSE_SIMD_N_PER_CHUNK - n - j);
//				m2Dst = __PULSE_SIMD_N_TO_ZEROS(m2Dst, __PULSE_SIMD_N_PER_CHUNK - n - j);
//			}
//			sum = __PULSE_SIMD_MADD(m1Dst, m2Dst ,sum);
//		}
//		dst[i] += __PULSE_SIMD_TO_FLOAT(__PULSE_SIMD_REDUCE_ADD(sum));
//	}
//}




static void __PULSE_SIMD_ADD_1x1(float * m1, float * m2, float * dst, int n)
{
	__PULSE_SIMD_BIGEST_DATA_TYPE m1Dst, m2Dst;
	for(int i = 0; i < n; i += __PULSE_SIMD_N_PER_CHUNK)
	{
		m1Dst = __PULSE_SIMD_LOAD(m1 + i);
		m2Dst = __PULSE_SIMD_LOAD(m2 + i);
		__PULSE_SIMD_GET(dst + i, __PULSE_SIMD_ADD(m1Dst, m2Dst));
	}
}




static void __PULSE_SIMD_MUL_1x1(float * m1, float * m2, float * dst, int n)
{
	__PULSE_SIMD_BIGEST_DATA_TYPE m1Dst, m2Dst, mul;
	for(int i = 0; i < n; i += __PULSE_SIMD_N_PER_CHUNK)
	{
		m1Dst = __PULSE_SIMD_LOAD(m1 + i);
		m2Dst = __PULSE_SIMD_LOAD(m2 + i);
		__PULSE_SIMD_GET(dst + i, __PULSE_SIMD_MUL(m1Dst, m2Dst));
	};
};



//static void __PULSE_MADD_1xM(float m, float * m1, float * dst, int n)
//{
//	__PULSE_SIMD_BIGEST_DATA_TYPE mDst, m1Dst, dstDst;
//	mDst = __PULSE_SIMD_SET_ALL(m);
//
//	for(int i = 0; i < n; i += __PULSE_SIMD_N_PER_CHUNK)
//	{
//		m1Dst = __PULSE_SIMD_LOAD(m1 + i);
//		dstDst = __PULSE_SIMD_LOAD(dst + i);
//		__PULSE_SIMD_GET(__PULSE_SIMD_MADD(mDst, m1Dst, dstDst), &dst[i]);
//	}
//}


//int main(int argc, char **argv) {
//
//	int R2 = 1024;
//	int K = 1024;
//	float * matrix_a = aligned_alloc(__PULSE_SIMD_CHUNK_SIZE, K*sizeof(float));
//	float * matrix_b = aligned_alloc(__PULSE_SIMD_CHUNK_SIZE, R2*K*sizeof(float));
//
//	float *result = malloc(K*sizeof(float));
//
//	for (int i = 0; i < K; i++) 
//		*(matrix_a+i) = 1;
//
//	for (int i = 0; i < R2*K; i++) 
//		*(matrix_b+i) = 1;
//
//	for (int i = 0; i < K; i++)
//		result[i] = 0;
//
//	double t1 = omp_get_wtime();
//	__PULSE_1MNxMMN(matrix_a, matrix_b, result, R2, K);
//	double t2 = omp_get_wtime();
//	printf("%f\n", t2 - t1);
//
//	for(int i = 0; i < K; i++)
//		printf("%.f", result[i]);
//
//	return 0;
//}
//
#endif
