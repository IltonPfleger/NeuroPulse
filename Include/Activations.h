#ifndef _PULSE_ACT
#define _PULSE_ACT
#include "PulseTypes.h"

typedef void (*PULSE_ActivationFunctionPtr)(PULSE_DataType *, size_t, char);
void* PULSE_GetActivationFunctionPtr(PULSE_ActivationFunction);

#endif
