#ifndef _PULSE_ACT
#define _PULSE_ACT
#include <math.h>
#include "PULSETypes.h"

static void PULSE_Softmax(PULSE_DataType*, PULSE_N, char);
static void PULSE_Sigmoid(PULSE_DataType*, PULSE_N, char);
static void PULSE_ReLU(PULSE_DataType*, PULSE_N, char);
void* PULSE_GetActivationFunctionPtr(PULSE_ActivationFunction);

#endif
