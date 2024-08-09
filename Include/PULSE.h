#ifndef _PULSE
#define _PULSE

#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "PULSETypes.h"
#include "Layer.h"
#include "Activations.h"
#include "Convolutional.h"
#include "MaxPool.h"
#include "Dense.h"

PULSE_DataType * PULSE_Foward(PULSE_Layer *, PULSE_DataType *);
void PULSE_Back(PULSE_Layer *);
void PULSE_Shuffle(PULSE_N *, PULSE_N);
void PULSE_Train(PULSE_Layer *, PULSE_N, PULSE_N, PULSE_HyperArgs, PULSE_DataType *, PULSE_DataType *);
void PULSE_Connect(PULSE_Layer *, PULSE_Layer * );
void PULSE_Destroy(PULSE_Layer *);

#endif
