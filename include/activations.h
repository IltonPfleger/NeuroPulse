#ifndef __ACTIVATIONS__
#define __ACTIVATIONS__

#include <pulse_types.h>

typedef enum {
    PULSE_ACTIVATION_NONE,
    PULSE_ACTIVATION_RELU,
    PULSE_ACTIVATION_LEAKYRELU,
    PULSE_ACTIVATION_SIGMOID,
} pulse_activation_fnc_e;

typedef void (*pulse_activation_fnc_ptr)(PULSE_DATA *, size_t, char);
void *pulse_get_activation_fnc_ptr(pulse_activation_fnc_e);

#endif
