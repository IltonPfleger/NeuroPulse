#ifndef _PULSE
#define _PULSE
#include "PulseTypes.h"
#include "Layer.h"
#include "Dense.h"
#include "Loss.h"
#include "Activations.h"

typedef struct
{
    PULSE_layer_t * layers;
    PULSE_data_t * weights;
    PULSE_data_t * io;
    size_t n_layers;
    size_t weights_size;
    size_t io_size;
    size_t fixes_size;
    size_t errors_size;
    char trained;
} PULSE_Model;

PULSE_data_t * PULSE_Foward(PULSE_layer_t *, PULSE_data_t *);
PULSE_Model PULSE_CreateModel(int, ...);
void PULSE_Destroy(PULSE_Model *);
void PULSE_Back(PULSE_layer_t *);
void PULSE_Shuffle(size_t *, size_t);
void PULSE_Train(PULSE_Model, size_t, size_t, PULSE_HyperArgs, PULSE_LossFunction, PULSE_data_t *, PULSE_data_t *);
void PULSE_Connect(PULSE_layer_t *, PULSE_layer_t * );

#endif
