#ifndef PULSE_ACTIVATION_RELU_H
#define PULSE_ACTIVATION_RELU_H

#include <activations/activation.h>
#include <types/dtype.h>

#define RELU(dtype)                                \
    static void relu_##dtype(void *ptr, int prime) \
    {                                              \
        dtype *data = (dtype *)ptr;                \
        if (prime) {                               \
            *data = *data > 0 ? 1 : 0;             \
        } else {                                   \
            *data = *data > 0 ? *data : 0;         \
        };                                         \
    };

RELU(int)
RELU(float)
RELU(double)

static const pulse_activation_t PULSE_RELU[] = {
    [PULSE_INT]    = relu_int,
    [PULSE_FLOAT]  = relu_float,
    [PULSE_DOUBLE] = relu_double,
};

#endif
