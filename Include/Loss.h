#ifndef _PULSE_LOSS
#define _PULSE_LOSS
#include <math.h>
#include "PULSETypes.h"

typedef void (*PULSE_LossFunctionPtr)(PULSE_DataType *, PULSE_DataType *, PULSE_DataType *, PULSE_N);
static void _MSE(PULSE_DataType *, PULSE_DataType *, PULSE_DataType *, PULSE_N);
static void _MAE(PULSE_DataType *, PULSE_DataType *, PULSE_DataType *, PULSE_N);
void* PULSE_GetLossFunctionPtr(PULSE_LossFunction);

#endif
