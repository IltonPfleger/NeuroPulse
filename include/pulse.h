#ifndef __PULSE__
#define __PULSE__

#include <layers/layer.h>
#include <losses/loss.h>
#include <stddef.h>
#include <types/train.h>

struct pulse_layer_s *pulse_create_model(int, ...);
void *pulse_forward(struct pulse_layer_s *, const void *const);
void pulse_free(struct pulse_layer_s *);
void pulse_back(struct pulse_layer_s *);
void pulse_train(struct pulse_layer_s *, pulse_train_args_t, pulse_loss_function, const void *const *, const void *const *);
void pulse_shuffle(size_t *, size_t);

#endif
