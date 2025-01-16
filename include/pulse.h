#pragma once
#include "pulse_types.h"
#include "layer.h"
#include "dense.h"
#include "loss.h"
#include "activations.h"


typedef struct {
    pulse_layer_t * layers;
    size_t n_layers;
} pulse_model;

PULSE_DATA * pulse_foward(pulse_layer_t *, PULSE_DATA *);
pulse_model pulse_create_model(int, ...);
void pulse_free(pulse_model *);
void pulse_back(pulse_layer_t *);
void pulse_shuffle(size_t *, size_t);
void pulse_train(pulse_model, size_t, size_t, pulse_train_hyper_args_t, pulse_loss_fnc_e, PULSE_DATA *, PULSE_DATA *);
