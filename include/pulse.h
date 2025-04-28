#ifndef __PULSE__
#define __PULSE__

#include <layers/layer.h>
#include <losses/loss.h>
#include <stddef.h>
#include <types/train.h>

typedef struct {
    pulse_layer_t *layers;
    size_t n_layers;
} pulse_model;

pulse_model pulse_create_model(int, ...);
void *pulse_forward(pulse_model, void *);
void pulse_free(pulse_model);
void pulse_back(pulse_model);
void pulse_shuffle(size_t *, size_t);
void pulse_train(pulse_model, pulse_train_args_t, pulse_loss_function, void **, void **);

#endif
