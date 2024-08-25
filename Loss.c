#include "Include/Loss.h"


static PULSE_data_t _MSE(PULSE_data_t * restrict x, PULSE_data_t * restrict y, PULSE_data_t * restrict errors, size_t size)
{
    PULSE_data_t loss = 0;
    for(int i = 0; i < size; i++)
        (errors[i] = 2*(x[i] - y[i]), loss += pow(x[i] - y[i], 2));
    return loss/size;
}

static PULSE_data_t _MSA(PULSE_data_t *restrict  x, PULSE_data_t *restrict  y, PULSE_data_t *restrict  errors, size_t size)
{
    PULSE_data_t loss = 0;
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



