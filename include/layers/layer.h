#ifndef PULSE_LAYER_H
#define PULSE_LAYER_H

#include <stddef.h>

typedef struct pulse_layer_s {
    void *inputs;
    void *outputs;
    void *errors;
    size_t isize;
    size_t osize;
    struct pulse_layer_s *next;
    struct pulse_layer_s *prev;
    void *internal;
    void (*forward)(struct pulse_layer_s *, const void *const);
    void (*think)(struct pulse_layer_s *);
    void (*free)(struct pulse_layer_s *);
    void (*learn)(struct pulse_layer_s *, size_t);
} pulse_layer_t;

#endif
