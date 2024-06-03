#ifndef _PULSE_CONV
#define _PULSE_CONV
#include "Layer.h"

typedef struct {
	int i_size[3];
	int o_size[3];
	int k_size;
	PULSE_DataType * gradients;
	PULSE_DataType * kernels;
	PULSE_DataType * baias;
	PULSE_DataType * deltas;
}PULSE_ConvolutionalLayer;


static void _FeedConvolutional(PULSE_Layer*);
static void _BackConvolutional(PULSE_Layer*);
static void _FixConvolutional(PULSE_Layer *, PULSE_HyperArgs);
static void _DestroyConvolutional(PULSE_Layer*);

PULSE_Layer PULSE_CreateConvolutionalLayer(int, int, int, int, int);

#endif

