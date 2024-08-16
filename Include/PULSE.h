#ifndef _PULSE
#define _PULSE
#include "PulseTypes.h"
#include "Layer.h"
#include "Dense.h"
#include "Loss.h"
#include "Activations.h"

typedef struct 
{
	PULSE_Layer * layers;
	PULSE_DataType * weights;
	PULSE_DataType * io;
	PULSE_DataType * fixes;
}PULSE_Model;

PULSE_DataType * PULSE_Foward(PULSE_Layer *, PULSE_DataType *);
PULSE_Model PULSE_CreateModel(int, ...);
void PULSE_Destroy(PULSE_Model *);
void PULSE_Back(PULSE_Layer *);
void PULSE_Shuffle(PULSE_N *, PULSE_N);
void PULSE_Train(PULSE_Layer *, PULSE_N, PULSE_N, PULSE_HyperArgs, PULSE_LossFunction, PULSE_DataType *, PULSE_DataType *);
void PULSE_Connect(PULSE_Layer *, PULSE_Layer * );

#endif
