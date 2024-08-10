#ifndef _PULSE_LOSS
#define _PULSE_LOSS
#include "PulseTypes.h"

typedef PULSE_DataType (*PULSE_LossFunctionPtr)(PULSE_DataType *, PULSE_DataType *, PULSE_DataType *, PULSE_N);
void* PULSE_GetLossFunctionPtr(PULSE_LossFunction);

#endif