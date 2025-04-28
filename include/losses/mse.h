#ifndef PULSE_MSE_H
#define PULSE_MSE_H

#include <losses/loss.h>
#include <math.h>
#include <types/dtype.h>

#define MSE(dtype)                                                                                  \
    static double mse_##dtype(const void* const x, const void* const y, void* const z, size_t size) \
    {                                                                                               \
        dtype* x_data = (dtype*)x;                                                                  \
        dtype* y_data = (dtype*)y;                                                                  \
        dtype* z_data = (dtype*)z;                                                                  \
        double loss   = 0;                                                                          \
        for (size_t i = 0; i < size; i++) {                                                         \
            dtype diff = x_data[i] - y_data[i];                                                     \
            z_data[i]  = 2 * diff;                                                                  \
            loss += pow(diff, 2);                                                                   \
        };                                                                                          \
        return loss;                                                                                \
    };

MSE(int)
MSE(float)
MSE(double)

static const pulse_loss_function PULSE_MSE[] = {
    [PULSE_INT]    = mse_int,
    [PULSE_FLOAT]  = mse_float,
    [PULSE_DOUBLE] = mse_double,
};

#endif
