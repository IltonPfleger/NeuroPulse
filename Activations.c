#include "Include/Activations.h"

static void PULSE_Sigmoid(PULSE_DataType * x, PULSE_N size, char prime)
{
	for(int i = 0; i < size; i++)
	{
		if(prime)
			x[i] = x[i]*(1.f - x[i]);
		x[i] = 1.f/(1.f + expf(-x[i]));
	}
}


static void PULSE_ReLU(PULSE_DataType * x, PULSE_N size, char prime)
{

	for(int i = 0; i < size; i++)
	{
		if(prime)
			x[i] = x[i] > 0 ? 1:0.0001;
		x[i] = x[i] > 0 ? x[i]:0.0001*x[i];
	}
}

void* PULSE_GetActivationFunctionPtr(PULSE_ActivationFunction type)
{
	switch(type)
	{
		case PULSE_ACTIVATION_NONE:
			break;
		case PULSE_ACTIVATION_SIGMOID:
			return PULSE_Sigmoid;
			break;
		case PULSE_ACTIVATION_RELU:
			return PULSE_ReLU;
			break;
	}
}



//void PULSE_Softmax(PULSE_DataType * x, PULSE_N size, char prime)
//{
//	if(prime)
//	{
//		for(int i = 0; i < size; i++)
//			x[i] = x[i]*(1.f - x[i]);
//		return;
//	}
//
//	double max = x[0];
//
//	for (int i = 1; i < size; i++)
//		if (x[i] > max) 
//			max = x[i];
//
//
//	double sum = 0;
//
//	for(int i = 0; i < size; i++)
//	{
//		x[i] = exp(x[i] - max);
//		sum += x[i];
//	}
//
//	for(int i = 0; i < size; i++)
//		x[i] /= sum;
//}
