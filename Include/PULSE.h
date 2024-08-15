#ifndef _PULSE
#define _PULSE
#include "PulseTypes.h"
#include "Layer.h"
#include "Dense.h"
#include "Loss.h"
#include "Activations.h"


PULSE_DataType * PULSE_Foward(PULSE_Layer *, PULSE_DataType *);
void PULSE_CreateModel(PULSE_Layer[], int, ...);
void PULSE_Back(PULSE_Layer *);
void PULSE_Shuffle(PULSE_N *, PULSE_N);
void PULSE_Train(PULSE_Layer *, PULSE_N, PULSE_N, PULSE_HyperArgs, PULSE_LossFunction, PULSE_DataType *, PULSE_DataType *);
void PULSE_Connect(PULSE_Layer *, PULSE_Layer * );
void PULSE_Destroy(PULSE_Layer *);

#endif
