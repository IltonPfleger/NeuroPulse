#ifndef _PULSE_ACT
#define _PULSE_ACT
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "PULSETypes.h"

typedef PULSE_Void (*PULSE_ActivationFunctionPtr)(PULSE_DataType *, PULSE_N, char);
void* PULSE_GetActivationFunctionPtr(PULSE_ActivationFunction);

#endif
