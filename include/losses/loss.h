#ifndef __LOSS__
#define __LOSS__

#include <types.h>

typedef pulse_datatype (*pulse_loss_function)(pulse_datatype *, pulse_datatype *, pulse_datatype *, size_t);

pulse_datatype pulse_mse(pulse_datatype *, pulse_datatype *, pulse_datatype *, size_t);
pulse_datatype pulse_msa(pulse_datatype *, pulse_datatype *, pulse_datatype *, size_t);

#endif
