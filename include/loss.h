#ifndef __LOSS__
#define __LOSS__

#include <pulse_types.h>

typedef PULSE_DATA (*pulse_loss_fnc_ptr)(PULSE_DATA *, PULSE_DATA *, PULSE_DATA *, size_t);
void *pulse_get_loss_fnc_ptr(pulse_loss_fnc_e);

#endif
