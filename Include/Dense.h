#ifndef _PULSE_DENSE
#define _PULSE_DENSE
#include "Layer.h"

typedef struct 
{
	unsigned int n_inputs;
	unsigned int n_outputs;
	PULSE_ActivationFunction activation_function;
	PULSE_OptimizationType optimization;
}PULSE_ARGS_DENSE;


typedef struct {
	PULSE_DataType * weights;
	PULSE_DataType * baiases;
	PULSE_DataType * deltas;
	PULSE_DataType * ddeltas;
	PULSE_DataType * gradients;
}PULSE_DenseLayer;

PULSE_Layer PULSE_CreateDenseLayer(int, int, PULSE_ActivationFunction, PULSE_OptimizationType);

#endif
