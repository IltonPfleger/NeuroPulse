#ifndef _PULSE_DENSE
#define _PULSE_DENSE
#include "Layer.h"

typedef struct
{
    unsigned int n_inputs;
    unsigned int n_outputs;
    PULSE_ActivationFunction activation_function;
    PULSE_OptimizationType optimization;
} PULSE_DenseLayerArgs;


PULSE_layer_t PULSE_CreateDenseLayer(PULSE_DenseLayerArgs);


#endif
