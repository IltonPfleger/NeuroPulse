#include "Include/Loss.h"


static PULSE_DataType _MSE(PULSE_DataType * x, PULSE_DataType * y, PULSE_DataType * errors, size_t size)
{
	PULSE_DataType loss = 0;
	for(int i = 0; i < size; i++)
		(errors[i] = 2*(x[i] - y[i]), loss += pow(x[i] - y[i], 2));
	return loss/size;
}

static PULSE_DataType _MSA(PULSE_DataType * x, PULSE_DataType * y, PULSE_DataType * errors, size_t size)
{
	PULSE_DataType loss = 0;
	for(int i = 0; i < size; i++)
		(errors[i] = fabs(x[i] - y[i]), loss += errors[i]);
	return loss/size;
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



