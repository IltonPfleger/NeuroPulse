#ifndef _PULSE_LAYER
#define _PULSE_LAYER
#include "PulseTypes.h"
#include "Activations.h"

typedef enum
{
    PULSE_DENSE,
} PULSE_layer_enum_t;


typedef struct PULSE_layer_t
{
    PULSE_data_t * inputs;
    PULSE_data_t * outputs;
    PULSE_data_t * errors;
    PULSE_data_t * w;
    PULSE_data_t * g;
    PULSE_layer_enum_t type;
    PULSE_OptimizationType optimization;
    PULSE_ActivationFunctionPtr activate;
    void (*feed)(struct PULSE_layer_t *);
    void (*back)(struct PULSE_layer_t *);
    void (*randomize)(struct PULSE_layer_t *);
    void (*start)(struct PULSE_layer_t *, PULSE_data_t **, PULSE_data_t **);
    void (*mode)(struct PULSE_layer_t *, PULSE_data_t **, PULSE_data_t **);
    struct PULSE_layer_t * parent;
    struct PULSE_layer_t * child;
    size_t n_inputs;
    size_t n_outputs;
} PULSE_layer_t;


#endif
