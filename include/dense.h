#pragma once
#include "layer.h"

typedef struct {
    unsigned int n_inputs;
    unsigned int n_outputs;
    pulse_activation_fnc_e activation_function;
    pulse_optimization_e optimization;
} pulse_dense_layer_args_t;

pulse_layer_t pulse_create_dense_layer(size_t, size_t, pulse_activation_fnc_e, pulse_optimization_e);
