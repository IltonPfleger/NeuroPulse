#pragma once
#include "pulse_types.h"
#include "activations.h"

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
    void (*randomize)(struct pulse_layer_s *);
    void (*start)(struct pulse_layer_s *, PULSE_DATA **, PULSE_DATA **);
    void (*mode)(struct pulse_layer_s *, PULSE_DATA **, PULSE_DATA **);
    struct pulse_layer_s * next;
    struct pulse_layer_s * prev;
    size_t n_inputs;
    size_t n_outputs;
} pulse_layer_t;


//Dense
typedef struct {
    unsigned int n_inputs;
    unsigned int n_outputs;
    pulse_activation_fnc_e activation_function;
    pulse_optimization_e optimization;
} pulse_dense_layer_args_t;

pulse_layer_t pulse_create_dense_layer(pulse_dense_layer_args_t);
