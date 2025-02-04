#ifndef __PULSE__
#define __PULSE__

#include <layers/layer.h>
#include <losses/loss.h>
#include <types.h>

typedef struct {
    pulse_layer_t *layers;
    size_t n_layers;
} pulse_model;

pulse_datatype *pulse_foward(pulse_layer_t *, pulse_datatype *);
pulse_model pulse_create_model(int, ...);
void pulse_free(pulse_model);
void pulse_back(pulse_layer_t *);
void pulse_shuffle(size_t *, size_t);
void pulse_train(pulse_model, size_t, size_t, pulse_train_hyper_args_t, pulse_loss_function, pulse_datatype *, pulse_datatype *);

#endif
