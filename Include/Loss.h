#ifndef _PULSE_LOSS
#define _PULSE_LOSS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "PULSETypes.h"

typedef void (*PULSE_LossFunctionPtr)(PULSE_DataType *, PULSE_DataType *, PULSE_DataType *, PULSE_N);
void* PULSE_GetLossFunctionPtr(PULSE_LossFunction);

#endif
