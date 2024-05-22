#ifndef NETWORK_H
#define NETWORK_H
#include "Layer.h"

void PULSE_Shuffle(int *indexes, int max);
void PULSE_Train(PULSE_Layer * input, double * x, double * y, int data_size, int batch_size, int epoch, double lr);
void PULSE_Connect(PULSE_Layer * parent, PULSE_Layer * child);

#endif
