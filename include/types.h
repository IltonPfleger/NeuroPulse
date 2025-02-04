#ifndef __TYPES__
#define __TYPES__

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef float pulse_datatype;

typedef struct {
    int batch_size;
    double lr;
} pulse_train_hyper_args_t;

#endif
