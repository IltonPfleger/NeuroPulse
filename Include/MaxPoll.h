#ifndef _PULSE_MAXPOLL
#define _PULSE_MAXPOLL
#include "Layer.h"


typedef struct {
	int i_size[3];
	int o_size[3];
	int k_size;
}PULSE_MaxPollLayer;

static void _FeedMaxPoll(PULSE_Layer *);
static void _BackMaxPoll(PULSE_Layer *);
static void _FixMaxPoll(PULSE_Layer *, PULSE_HyperArgs);
static void _DestroyMaxPoll(PULSE_Layer *);
PULSE_Layer PULSE_CreateMaxPollLayer(int, int, int, int);

#endif
