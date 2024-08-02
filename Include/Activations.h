#ifndef _PULSE_ACT
#define _PULSE_ACT
#include <math.h>
#include <omp.h>
#include "Layer.h"

void PULSE_Softmax(PULSE_Layer * layer, char prime);
void PULSE_Sigmoid(PULSE_Layer * layer, char prime);
void PULSE_ReLU(PULSE_Layer * layer, char prime);

#endif
