#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdalign.h>

#define __PLA_MIN(x, y) ((x) < (y) ? (x) : (y))
#define __PLA_MAX(x, y) ((x) > (y) ? (x) : (y))

#if defined(__AVX512F__)
#define __PLA_SIMD_CHUNK_SIZE 512
#elif defined(__AVX__) || defined(__AVX2__)
#define __PLA_SIMD_CHUNK_SIZE 256
#elif defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__)
#define __PLA_SIMD_CHUNK_SIZE 128
#else
#define __PLA_SIMD_CHUNK_SIZE 0
#endif
#define __PLA_FLOAT_PS_SIZE 32
#define __PLA_FLOAT_PD_SIZE 64
#define __PLA_SIMD_N_PER_CHUNK __PLA_SIMD_CHUNK_SIZE/__PLA_FLOAT_PS_SIZE


#if defined(__i386__) || defined(__x86_64__)
#define __PLA_ARCH 0
#include <immintrin.h>
static inline __m128 __PLA_SIMD_X86_REDUCE_ADD_128(__m128 x) { __m128 tmp = _mm_hadd_ps(x, x); return _mm_hadd_ps(tmp, tmp);};
static inline __m128 __PLA_SIMD_X86_REDUCE_ADD_256(__m256 x) { __m128 tmp = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)); tmp = _mm_hadd_ps(tmp, tmp); return _mm_hadd_ps(tmp, tmp);};
//static inline __m128 __PLA_SIMD_X86_N_TO_ZEROS_128(__m128 x, int n) { float m[4]; for(int i = 0; i < 4; i++)m[i] = i < n ? 1 : -1; __m128 mf = _mm_loadu_ps(m); return _mm_blendv_ps(x, _mm_set1_ps(0.0f), mf);};
static inline __m256 __PLA_SIMD_X86_N_TO_ZEROS_256(__m256 x, int n) { float m[8]; for(int i = 0; i < 8; i++)m[i] = i < n ? 1 : -1; __m256 mf = _mm256_loadu_ps(m); return _mm256_blendv_ps(x, _mm256_set1_ps(0.0f), mf);};
#endif


#if __PLA_ARCH == 0 && __PLA_SIMD_CHUNK_SIZE == 256
#define __PLA_SIMD_BIGEST_DATA_TYPE __m256
#define __PLA_SIMD_LOWEST_DATA_TYPE __m128
#define __PLA_SIMD_LOAD(x) _mm256_loadu_ps(x)
#define __PLA_SIMD_MUL(x, y) _mm256_mul_ps(x, y)
#define __PLA_SIMD_ADD(x, y) _mm256_add_ps(x, y)
#define __PLA_SIMD_ZERO() _mm256_setzero_ps()
#define __PLA_SIMD_SET_ALL(x) _mm256_set1_ps(x)
#define __PLA_SIMD_GET(x, dst) _mm256_storeu_ps(dst, x)
#define __PLA_SIMD_REDUCE_ADD(x) __PLA_SIMD_X86_REDUCE_ADD_256(x)
#define __PLA_SIMD_TO_FLOAT(x) _mm_cvtss_f32(x)
#define __PLA_SIMD_N_TO_ZEROS(x, n) __PLA_SIMD_X86_N_TO_ZEROS_256(x, n)
#define __PLA_SIMD_MADD(x, y, z) _mm256_fmadd_ps(x, y, z)
#endif

#if __PLA_ARCH == 0 && __PLA_SIMD_CHUNK_SIZE == 128
#define __PLA_SIMD_BIGEST_DATA_TYPE __m128
#define __PLA_SIMD_LOWEST_DATA_TYPE __m128
#define __PLA_SIMD_LOAD(x) _mm_loadu_ps(x)
#define __PLA_SIMD_MUL(x, y) _mm_mul_ps(x, y)
#define __PLA_SIMD_ADD(x, y) _mm_add_ps(x, y)
#define __PLA_SIMD_ZERO() _mm_setzero_ps()
#define __PLA_SIMD_GET(x, dst) _mm_storeu_ps(dst, x)
#define __PLA_SIMD_REDUCE_ADD(x) __PLA_SIMD_X86_REDUCE_ADD_128(x)
#define __PLA_SIMD_TO_FLOAT(x) _mm_cvtss_f32(x)
//#define __PLA_SIMD_N_TO_ZEROS(x, n) __PLA_SIMD_X86_N_TO_ZEROS_128(x, n)
#endif

//static void __PLA_MxM(float * m1, float * m2, float * dst, int m1R, int m2R, int K)
//{
//	__PLA_SIMD_BIGEST_DATA_TYPE rM1, rM2, simd_mul;
//	__PLA_SIMD_LOWEST_DATA_TYPE simd_sum;
//
////#pragma omp parallel for
//	for (int i = 0; i < m1R; i++)
//	{
//		for (int j = 0; j < m2R; j++)
//		{
//			int k = 0;
//			float acc = 0;
//			for (; k < K; k += __PLA_SIMD_N_PER_CHUNK)
//			{
//				rM1 = __PLA_SIMD_LOAD(m1+(i*K)+k);
//				rM2 = __PLA_SIMD_LOAD(m2+(j*K)+k);
//				simd_mul = __PLA_SIMD_MUL(rM1, rM2);
//				simd_sum = __PLA_SIMD_REDUCE_ADD(simd_mul);
//				acc += __PLA_SIMD_TO_FLOAT(simd_sum);
//			}
//			if(k < K)
//			{
//				for(; k < K; k++)
//				{
//					acc += m1[i * K + k] * m2[j * K + k];
//				}
//			}
//			dst[i * m1R + j] = acc;
//		}
//	}
//}

//static void __PLA_MxTM(float * m1, float * m2, float * dst, int m1R, int m2R, int K)
//{
//	__PLA_SIMD_BIGEST_DATA_TYPE localM1[__PLA_SIMD_N_PER_CHUNK];
//	__PLA_SIMD_BIGEST_DATA_TYPE localM2[__PLA_SIMD_N_PER_CHUNK];
//	__PLA_SIMD_BIGEST_DATA_TYPE simd_mul;
//	__PLA_SIMD_LOWEST_DATA_TYPE simd_sum;
//	alignas(__PLA_SIMD_N_PER_CHUNK * __PLA_FLOAT_PS_SIZE) float localDst[__PLA_SIMD_N_PER_CHUNK][__PLA_SIMD_N_PER_CHUNK];
//	const __PLA_SIMD_BIGEST_DATA_TYPE zero = __PLA_SIMD_ZERO();
//	const int R1 = m1R%__PLA_SIMD_N_PER_CHUNK == 0 ? m1R : m1R + __PLA_SIMD_N_PER_CHUNK;
//	const int R2 = m2R%__PLA_SIMD_N_PER_CHUNK == 0 ? m2R : m2R + __PLA_SIMD_N_PER_CHUNK;
//
//
//#pragma omp parallel for private(simd_mul, simd_sum, localM1, localM2, localDst) 
//	for(int i = 0; i < R1; i += __PLA_SIMD_N_PER_CHUNK)
//	{
//		const int TR1 = __PLA_MIN(m1R - i, __PLA_SIMD_N_PER_CHUNK);
//		for(int j = 0; j < R2; j += __PLA_SIMD_N_PER_CHUNK)
//		{
//			const int TR2 = __PLA_MIN(m2R - j, __PLA_SIMD_N_PER_CHUNK);
//			for(int k = 0; k < K; k += __PLA_SIMD_N_PER_CHUNK)
//			{
//				for(int l = 0; l < __PLA_SIMD_N_PER_CHUNK; l++)
//				{
//					localM1[l] = __PLA_SIMD_LOAD(m1 + (i + l)*K + k);
//					localM2[l] = __PLA_SIMD_LOAD(m2 + (j + l)*K + k);
//					for(int m = 0; m < __PLA_SIMD_N_PER_CHUNK; m++)
//						localDst[l][m] = 0.f;
//					if(k + __PLA_SIMD_N_PER_CHUNK > K)
//					{
//						localM1[l] = __PLA_SIMD_N_TO_ZEROS(localM1[l], K - k);
//						localM2[l] = __PLA_SIMD_N_TO_ZEROS(localM2[l], K - k);
//					}
//				}
//				for(int l = 0; l < TR1; l++)
//				{
//					for(int m = 0; m < TR2; m++)
//					{
//						simd_mul = __PLA_SIMD_MUL(localM1[l], localM2[m]);
//						simd_sum = __PLA_SIMD_REDUCE_ADD(simd_mul);
//						localDst[l][m] += __PLA_SIMD_TO_FLOAT(simd_sum);
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
//
//static void __PLA_1MNxMMN(float * m1, float * m2, float * dst, int n, int m)
//{
//	__PLA_SIMD_BIGEST_DATA_TYPE m1Dst, m2Dst, sum;
//#pragma omp parallel for schedule(static) private(m1Dst, m2Dst, sum)
//	for(int i = 0; i < m; i++)
//	{
//		const int wi = i*n;
//		sum = __PLA_SIMD_ZERO();
//		for(int j = 0; j < n; j += __PLA_SIMD_N_PER_CHUNK)
//		{
//			m2Dst = __PLA_SIMD_LOAD(m2 + wi + j);
//			m1Dst = __PLA_SIMD_LOAD(m1 + j);
//			if(j + __PLA_SIMD_N_PER_CHUNK > n)
//			{
//				m1Dst = __PLA_SIMD_N_TO_ZEROS(m1Dst, __PLA_SIMD_N_PER_CHUNK - n - j);
//				m2Dst = __PLA_SIMD_N_TO_ZEROS(m2Dst, __PLA_SIMD_N_PER_CHUNK - n - j);
//			}
//			sum = __PLA_SIMD_MADD(m1Dst, m2Dst ,sum);
//		}
//		dst[i] = __PLA_SIMD_TO_FLOAT(__PLA_SIMD_REDUCE_ADD(sum));
//	}
//}
//
//
//static void __PLA_ADD_1MNxMMN(float * m1, float * m2, float * dst, int n, int m)
//{
//	__PLA_SIMD_BIGEST_DATA_TYPE m1Dst, m2Dst, sum;
//#pragma omp parallel for schedule(static) private(m1Dst, m2Dst, sum)
//	for(int i = 0; i < m; i++)
//	{
//		const int wi = i*n;
//		sum = __PLA_SIMD_ZERO();
//		for(int j = 0; j < n; j += __PLA_SIMD_N_PER_CHUNK)
//		{
//			m2Dst = __PLA_SIMD_LOAD(m2 + wi + j);
//			m1Dst = __PLA_SIMD_LOAD(m1 + j);
//			if(j + __PLA_SIMD_N_PER_CHUNK > n)
//			{
//				m1Dst = __PLA_SIMD_N_TO_ZEROS(m1Dst, __PLA_SIMD_N_PER_CHUNK - n - j);
//				m2Dst = __PLA_SIMD_N_TO_ZEROS(m2Dst, __PLA_SIMD_N_PER_CHUNK - n - j);
//			}
//			sum = __PLA_SIMD_MADD(m1Dst, m2Dst ,sum);
//		}
//		dst[i] += __PLA_SIMD_TO_FLOAT(__PLA_SIMD_REDUCE_ADD(sum));
//	}
//}
//
//static void __PLA_ADD_1x1(float * m1, float * m2, float * dst, int n)
//{
//	__PLA_SIMD_BIGEST_DATA_TYPE m1Dst, m2Dst;
//	for(int i = 0; i < n; i += __PLA_SIMD_N_PER_CHUNK)
//	{
//		m1Dst = __PLA_SIMD_LOAD(m1 + i);
//		m2Dst = __PLA_SIMD_LOAD(m2 + i);
//		__PLA_SIMD_GET(__PLA_SIMD_ADD(m1Dst, m2Dst), &dst[i]);
//	}
//}
//
//
//static void __PLA_MUL_1x1(float * m1, float * m2, float * dst, int n)
//{
//	__PLA_SIMD_BIGEST_DATA_TYPE m1Dst, m2Dst, mul;
//	for(int i = 0; i < n; i += __PLA_SIMD_N_PER_CHUNK)
//	{
//		m1Dst = __PLA_SIMD_LOAD(m1 + i);
//		m2Dst = __PLA_SIMD_LOAD(m2 + i);
//		__PLA_SIMD_GET(__PLA_SIMD_MUL(m1Dst, m2Dst), dst + i);
//	};
//};
//
//static void __PLA_MADD_1xM(float m, float * m1, float * dst, int n)
//{
//	__PLA_SIMD_BIGEST_DATA_TYPE mDst, m1Dst, dstDst;
//	mDst = __PLA_SIMD_SET_ALL(m);
//
//	for(int i = 0; i < n; i += __PLA_SIMD_N_PER_CHUNK)
//	{
//		m1Dst = __PLA_SIMD_LOAD(m1 + i);
//		dstDst = __PLA_SIMD_LOAD(dst + i);
//		__PLA_SIMD_GET(__PLA_SIMD_MADD(mDst, m1Dst, dstDst), &dst[i]);
//	}
//}
//
//
//
//
//
//
//
//
////int main(int argc, char **argv) {
////	int R1 = 8;
////	int R2 = 8;
////	int K = 5;
////	float * matrix_a = aligned_alloc(__PLA_SIMD_CHUNK_SIZE, R1*K*sizeof(float));
////	float * matrix_b = aligned_alloc(__PLA_SIMD_CHUNK_SIZE, R2*K*sizeof(float));
////
////	float *result = malloc(R1*R2*sizeof(float));
////
////	for (int i = 0; i < R1*K; i++) 
////		*(matrix_a+i) = 1;
////
////	for (int i = 0; i < R2*K; i++) 
////		*(matrix_b+i) = 1;
////
////	for (int i = 0; i < R1; i++) {
////		for (int j = 0; j < R2; j++) {
////			result[i * R2 + j] = 0;
////		}
////	}
////
////	double t1 = omp_get_wtime();
////	__PLA_MxTM(matrix_a, matrix_b, result, R1, R2, K);
////	double t2 = omp_get_wtime();
////	printf("%f\n", t2 - t1);
////
////	for(int i = 0; i < R1; i++)
////	{
////		for(int j = 0; j < R2; j++)
////			printf("%.f ", result[i * R2 + j]);
////		printf("\n");
////	}
////
////	return 0;
////}
//
