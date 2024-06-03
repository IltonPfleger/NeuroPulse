#ifndef _PULSE
#define _PULSE

#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "Layer.h"
#include "Activations.h"
#include "Convolutional.h"
#include "MaxPoll.h"
#include "Dense.h"

void PULSE_Foward(PULSE_Layer * layer, PULSE_DataType * inputs);
void PULSE_Back(PULSE_Layer * layer);
void PULSE_Shuffle(int *indexes, int max);
void PULSE_Train(PULSE_Layer * first_layer, int epoch, int data_size, PULSE_HyperArgs args, PULSE_DataType * x, PULSE_DataType * y);
void PULSE_Connect(PULSE_Layer * parent, PULSE_Layer * child);
void PULSE_Destroy(PULSE_Layer * layer);

#endif
