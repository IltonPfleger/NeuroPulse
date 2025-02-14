#ifndef PULSE_LAYER_H
#define PULSE_LAYER_H

#include <types/train.h>

typedef struct pulse_layer_s {
    void *inputs;
    void *outputs;
    void *errors;
    size_t isize;
    size_t osize;
    struct pulse_layer_s *next;
    struct pulse_layer_s *prev;
    void *internal;
    void (*feed)(struct pulse_layer_s *, void *);
    void (*back)(struct pulse_layer_s *);
    void (*free)(struct pulse_layer_s *);
    void (*fix)(struct pulse_layer_s *, pulse_train_args_t);
} pulse_layer_t;

#endif
