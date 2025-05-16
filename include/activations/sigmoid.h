#ifndef PULSE_ACTIVATION_SIGMOID_H
#define PULSE_ACTIVATION_SIGMOID_H

#include <activations/activation.h>
#include <math.h>
#include <types/dtype.h>

#define SIGMOID(dtype)                                \
    static void sigmoid_##dtype(void *ptr, int prime) \
    {                                                 \
        dtype *data = (dtype *)ptr;                   \
        if (prime) {                                  \
            *data = *data * (1.f - *data);            \
        } else {                                      \
            *data = 1.f / (1.f + expf(-*data));       \
        };                                            \
    };

SIGMOID(float)
SIGMOID(double)

static const pulse_activation_t PULSE_SIGMOID[] = {
    [PULSE_INT]    = NULL,
    [PULSE_FLOAT]  = sigmoid_float,
    [PULSE_DOUBLE] = sigmoid_double,
};

#endif
