#ifndef _PULSE_ACT
#define _PULSE_ACT
#include "PulseTypes.h"

typedef PULSE_Void (*PULSE_ActivationFunctionPtr)(PULSE_DataType *, PULSE_Size_t, char);
void* PULSE_GetActivationFunctionPtr(PULSE_ActivationFunction);

#endif
