#ifndef __PULSE_TYPES__
#define __PULSE_TYPES__
#define PULSE_DataType float
#define PULSE_N unsigned int
#define PULSE_Void void

typedef struct {
	int batch_size;
	double lr;
} PULSE_HyperArgs; 

typedef enum 
{
	PULSE_OPTIMIZATION_NONE,
	PULSE_OPTIMIZATION_SIMD,
	PULSE_OPTIMIZATION_GPU_OPENCL,
} PULSE_OptimizationType;


#endif


