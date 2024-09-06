#pragma once
#include "pulse_types.h"

typedef void (*pulse_activation_fnc_ptr)(PULSE_DATA *, size_t, char);
void * pulse_get_activation_fnc_ptr(pulse_activation_fnc_e);
