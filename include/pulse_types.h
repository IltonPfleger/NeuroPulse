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
#define PULSE_ALLOC(x) aligned_alloc(__PULSE_CFLAGS_CacheLineSize, x)
#define PULSE_FREE(x) free(x)



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
} pulse_optimization_e;


#endif


