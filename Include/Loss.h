#ifndef _PULSE_LOSS
#define _PULSE_LOSS
#include "PulseTypes.h"

typedef PULSE_data_t (*PULSE_LossFunctionPtr)(PULSE_data_t *, PULSE_data_t *, PULSE_data_t *, size_t);
void* PULSE_GetLossFunctionPtr(PULSE_LossFunction);

#endif
