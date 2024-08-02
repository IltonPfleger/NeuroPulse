#ifndef _PULSE_MAXPOOL
#define _PULSE_MAXPOOL
#include "Layer.h"


typedef struct {
	PULSE_N i_size[3];
	PULSE_N o_size[3];
	PULSE_N k_size;
}PULSE_MaxPoolLayer;

static void _FeedMaxPool(PULSE_Layer *);
static void _BackMaxPool(PULSE_Layer *);
static void _FixMaxPool(PULSE_Layer *, PULSE_HyperArgs);
static void _DestroyMaxPool(PULSE_Layer *);
PULSE_Layer PULSE_CreateMaxPoolLayer(PULSE_N, PULSE_N, PULSE_N, PULSE_N);

#endif
