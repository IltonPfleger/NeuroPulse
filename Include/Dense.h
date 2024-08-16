#ifndef _PULSE_DENSE
#define _PULSE_DENSE
#include "Layer.h"

typedef struct 
{
	unsigned int n_inputs;
	unsigned int n_outputs;
	PULSE_ActivationFunction activation_function;
	PULSE_OptimizationType optimization;
}PULSE_DenseLayerArgs;


PULSE_Layer PULSE_CreateDenseLayer(PULSE_DenseLayerArgs, PULSE_DataType *, PULSE_DataType *);
unsigned int PULSE_GetDenseWeightsSize(PULSE_DenseLayerArgs);
unsigned int PULSE_GetDenseIOSize(PULSE_DenseLayerArgs);

#endif
