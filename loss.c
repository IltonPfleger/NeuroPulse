#include "loss.h"


static PULSE_DATA _MSE(PULSE_DATA * restrict x, PULSE_DATA * restrict y, PULSE_DATA * restrict errors, size_t size)
{
    PULSE_DATA loss = 0;
    for(int i = 0; i < size; i++)
        (errors[i] = 2*(x[i] - y[i]), loss += pow(x[i] - y[i], 2));
    return loss/size;
}

static PULSE_DATA _MSA(PULSE_DATA *restrict  x, PULSE_DATA *restrict  y, PULSE_DATA *restrict  errors, size_t size)
{
    PULSE_DATA loss = 0;
    for(int i = 0; i < size; i++)
        (errors[i] = fabs(x[i] - y[i]), loss += errors[i]);
    return loss/size;
}


void * pulse_get_loss_fnc_ptr(pulse_loss_fnc_e type)
{
    switch(type) {
        case PULSE_LOSS_MSE:
            return _MSE;
            break;
        case PULSE_LOSS_MAE:
            return _MSA;
            break;
    }
    return NULL;
}



