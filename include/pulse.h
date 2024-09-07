#pragma once
#include "pulse_types.h"
#include "layer.h"
#include "dense.h"
#include "loss.h"
#include "activations.h"

typedef struct {
    pulse_layer_t * layers;
    PULSE_DATA * weights;
    PULSE_DATA * io;
    size_t n_layers;
    size_t weights_size;
    size_t io_size;
    size_t fixes_size;
    size_t errors_size;
} pulse_model;

PULSE_DATA * pulse_foward(pulse_layer_t *, PULSE_DATA *);
pulse_model pulse_create_model(int, ...);
void pulse_destroy(pulse_model *);
void pulse_back(pulse_layer_t *);
void pulse_shuffle(size_t *, size_t);
void pulse_train(pulse_model, size_t, size_t, pulse_train_hyper_args_t, pulse_loss_fnc_e, PULSE_DATA *, PULSE_DATA *);
