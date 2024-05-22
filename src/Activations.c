#include "Activations.h"
#include <math.h>

void PULSE_Sigmoid(double * x, int size, int prime)
{
	for(int i = 0; i < size; i++)
	{
		if(prime)
			x[i] = x[i]*(1.f - x[i]);
		x[i] = 1.f/(1.f + expf(-x[i]));
	}
}

void PULSE_Softmax(double *x, int size, int prime)
{
	if(prime)
	{
		for(int i = 0; i < size; i++)
			x[i] = x[i]*(1 - x[i]);
		return;
	}

	double max = x[0];
	for (int i = 1; i < size; i++)
		if (x[i] > max) 
			max = x[i];


	double sum = 0;
	for(int i = 0; i < size; i++)
	{
		x[i] = exp(x[i] - max);
		sum += x[i];
	}

	for(int i = 0; i < size; i++)
		x[i] /= sum;
}


void PULSE_ReLU(double *x, int size, int prime)
{
	for(int i = 0; i < size; i++)
	{
		if(prime)
			x[i] = x[i] > 0 ? 1 : 0.0001;
		x[i] = x[i] > 0 ? x[i]: x[i]*0.0001;
	}
}


