#ifndef _PULSE
#define _PULSE
#include "PulseTypes.h"
#include "Layer.h"
#include "Dense.h"
#include "Loss.h"
#include "Activations.h"

typedef struct
{
    PULSE_Layer * layers;
    PULSE_DataType * weights;
    PULSE_DataType * io;
    size_t n_layers;
    size_t weights_size;
    size_t io_size;
    size_t fixes_size;
    size_t errors_size;
} PULSE_Model;

PULSE_DataType * PULSE_Foward(PULSE_Layer *, PULSE_DataType *);
PULSE_Model PULSE_CreateModel(int, ...);
void PULSE_Destroy(PULSE_Model *);
void PULSE_Back(PULSE_Layer *);
void PULSE_Shuffle(size_t *, size_t);
void PULSE_Train(PULSE_Model, size_t, size_t, PULSE_HyperArgs, PULSE_LossFunction, PULSE_DataType *, PULSE_DataType *);
void PULSE_Connect(PULSE_Layer *, PULSE_Layer * );

#endif
