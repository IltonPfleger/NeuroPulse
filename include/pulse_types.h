#ifndef __PULSE_TYPES__
#define __PULSE_TYPES__
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define PULSE_DATA float

typedef struct {
    int batch_size;
    double lr;
} pulse_train_hyper_args_t;

typedef enum {
    PULSE_LOSS_MSE,
    PULSE_LOSS_MAE,
} pulse_loss_fnc_e;

typedef enum {
    PULSE_OPTIMIZATION_NONE,
    PULSE_OPTIMIZATION_SIMD,
    PULSE_OPTIMIZATION_GPU_OPENCL,
} pulse_optimization_e;


#endif


