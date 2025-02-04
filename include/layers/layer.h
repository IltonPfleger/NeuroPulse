#ifndef __LAYER__
#define __LAYER__

#include <types.h>

typedef enum {
    PULSE_DENSE,
} pulse_layer_e;

typedef struct pulse_layer_s {
    pulse_datatype *inputs;
    pulse_datatype *outputs;
    pulse_datatype *errors;
    size_t n_inputs;
    size_t n_outputs;
    struct pulse_layer_s *next;
    struct pulse_layer_s *prev;
    void *internal;
    void (*feed)(struct pulse_layer_s *);
    void (*back)(struct pulse_layer_s *);
    void (*free)(struct pulse_layer_s *);
    void (*fix)(struct pulse_layer_s *, double);
} pulse_layer_t;

#endif
