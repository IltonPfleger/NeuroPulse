#ifndef __ACTIVATIONS__
#define __ACTIVATIONS__

#include <types.h>

typedef void (*pulse_activation_function)(pulse_datatype *, size_t, char);

void pulse_sigmoid(pulse_datatype *, size_t, char);
void pulse_relu(pulse_datatype *, size_t, char);
void pulse_leaky_relu(pulse_datatype *, size_t, char);

#endif
