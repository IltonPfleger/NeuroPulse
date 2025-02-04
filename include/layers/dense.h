#ifndef __DENSE__
#define __DENSE__

#include <layers/layer.h>

typedef struct {
    pulse_datatype* w;
    pulse_datatype* b;
    pulse_datatype* g;
    pulse_datatype* d;
    pulse_activation_function activate;
} pulse_dense_layer_t;

pulse_layer_t pulse_create_dense_layer(size_t, size_t, pulse_activation_function);

#endif
