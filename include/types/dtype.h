#ifndef PULSE_TYPES_DTYPE_H
#define PULSE_TYPES_DTYPE_H

#include <stddef.h>

typedef enum { PULSE_INT = 0, PULSE_FLOAT, PULSE_DOUBLE } pulse_dtype_t;

static const size_t pulse_dtype_sizes[] = {[PULSE_INT] = sizeof(int), [PULSE_FLOAT] = sizeof(float), [PULSE_DOUBLE] = sizeof(double)};

#endif
