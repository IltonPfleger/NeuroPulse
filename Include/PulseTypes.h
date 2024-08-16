#ifndef __PULSE_TYPES__
#define __PULSE_TYPES__
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define PULSE_DataType float
#define PULSE_Size_t unsigned int
#define PULSE_Void void

typedef struct {
	int batch_size;
	double lr;
} PULSE_HyperArgs; 

typedef enum
{
	PULSE_ACTIVATION_NONE,
	PULSE_ACTIVATION_RELU,
	PULSE_ACTIVATION_LEAKYRELU,
	PULSE_ACTIVATION_SIGMOID,
	PULSE_ACTIVATION_SOFTMAX,
}PULSE_ActivationFunction;

typedef enum
{
	PULSE_LOSS_MSE,
	PULSE_LOSS_MAE,
}PULSE_LossFunction;

typedef enum 
{
	PULSE_OPTIMIZATION_NONE,
	PULSE_OPTIMIZATION_SIMD,
	PULSE_OPTIMIZATION_GPU_OPENCL,
} PULSE_OptimizationType;


#endif


