#ifndef PULSE_L1_H
#define PULSE_L1_H

#include <losses/loss.h>
#include <math.h>
#include <types/dtype.h>

#define L1(dtype)                                                                                  \
    static double l1_##dtype(const void *const x, const void *const y, void *const z, size_t size) \
    {                                                                                              \
        dtype *x_data = (dtype *)x;                                                                \
        dtype *y_data = (dtype *)y;                                                                \
        dtype *z_data = (dtype *)z;                                                                \
        double loss   = 0;                                                                         \
        for (size_t i = 0; i < size; i++) {                                                        \
            dtype diff = x_data[i] - y_data[i];                                                    \
            z_data[i]  = diff;                                                                     \
            loss += diff;                                                                          \
        };                                                                                         \
        return loss;                                                                               \
    };

L1(int)
L1(float)
L1(double)

static const pulse_loss_function PULSE_L1[] = {
    [PULSE_INT]    = l1_int,
    [PULSE_FLOAT]  = l1_float,
    [PULSE_DOUBLE] = l1_double,
};

#endif
