#include "Include/PulseAlloc.h"

void * PULSE_Alloc(int aligned_size, int size, int x, PULSE_OptimizationType optimization)
{
	switch(optimization)
	{
		case PULSE_OPTIMIZATION_NONE:
			return aligned_alloc(__PULSE_CFLAGS_CacheLineSize, size*x);
			break;
		case PULSE_OPTIMIZATION_SIMD:
			return __PULSE_SIMD_CHECK(aligned_alloc(__PULSE_CFLAGS_CacheLineSize, size*(x%__PULSE_SIMD_N_PER_CHUNK == 0 ? x : x + x%__PULSE_SIMD_N_PER_CHUNK)));
			break;
		case PULSE_OPTIMIZATION_GPU_OPENCL:
			break;
	}
	exit(1);
	return NULL;
}


void * PULSE_Alloc2D(int aligned_size, int size, int x, int y, PULSE_OptimizationType optimization)
{
	switch(optimization)
	{
		case PULSE_OPTIMIZATION_NONE:
			return aligned_alloc(__PULSE_CFLAGS_CacheLineSize, size*x*y);
			break;
		case PULSE_OPTIMIZATION_SIMD:
			const int pad_x = x%__PULSE_SIMD_N_PER_CHUNK == 0 ? x : x + x%__PULSE_SIMD_N_PER_CHUNK;
			return __PULSE_SIMD_CHECK(aligned_alloc(__PULSE_CFLAGS_CacheLineSize, size*pad_x*y));
			break;
		case PULSE_OPTIMIZATION_GPU_OPENCL:
			break;
	}
	exit(1);
	return NULL;
}

void PULSE_Free(void * ptr) {free(ptr);};
