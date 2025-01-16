#pragma once
#include "pulse_types.h"
#include "activations.h"

typedef enum {
    PULSE_DENSE,
} pulse_layer_e;

typedef struct pulse_layer_s {
    PULSE_DATA * inputs;
    PULSE_DATA * outputs;
    PULSE_DATA * errors;
    PULSE_DATA * w;
    PULSE_DATA * g;
    pulse_layer_e type;
    pulse_optimization_e optimization;
    pulse_activation_fnc_ptr activate;
    void (*feed)(struct pulse_layer_s *);
    void (*back)(struct pulse_layer_s *);
    void (*free)(struct pulse_layer_s *);
    struct pulse_layer_s * next;
    struct pulse_layer_s * prev;
    size_t n_inputs;
    size_t n_outputs;
    size_t n_weights;
} pulse_layer_t;
