#include "Include/Loss.h"


static void _MSE(PULSE_DataType * x, PULSE_DataType * y, PULSE_DataType * errors, PULSE_N size)
{
	for(int i = 0; i < size; i++)
		errors[i] = 2*(x[i] - y[i]);
}

static void _MSA(PULSE_DataType * x, PULSE_DataType * y, PULSE_DataType * errors, PULSE_N size)
{
	for(int i = 0; i < size; i++)
		errors[i] = fabs(x[i] - y[i]);
}


void* PULSE_GetLossFunctionPtr(PULSE_LossFunction type)
{
	switch(type)
	{
		case PULSE_LOSS_MSE:
			return _MSE;
			break;
		case PULSE_LOSS_MAE:
			return _MSA;
			break;
	}
	return NULL;
}



