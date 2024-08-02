#ifndef _PULSE_DENSE
#define _PULSE_DENSE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Layer.h"

typedef struct {
	PULSE_DataType * weights;
	PULSE_DataType * baias;
	PULSE_DataType * deltas;
	PULSE_DataType * gradients;
}PULSE_DenseLayer;

static void _DestroyDense(PULSE_Layer *);
static void _FeedDense(PULSE_Layer *);
static void _BackDense(PULSE_Layer *);
static void _FixDense(PULSE_Layer *, PULSE_HyperArgs);
PULSE_Layer PULSE_CreateDenseLayer(int, int, PULSE_ActivationLayerFunctionPtr);

#endif
