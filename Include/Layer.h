#ifndef _PULSE_LAYER
#define _PULSE_LAYER
#include <stdio.h>
#include <stdlib.h>
#include "PULSETypes.h"
#include "Activations.h"

struct PULSE_Layer;
typedef PULSE_Void (*PULSE_FeedLayerFunctionPtr)(struct PULSE_Layer *);
typedef PULSE_Void (*PULSE_BackLayerFunctionPtr)(struct PULSE_Layer *);
typedef PULSE_Void (*PULSE_FixLayerFunctionPtr)(struct PULSE_Layer *, PULSE_HyperArgs);
typedef PULSE_Void (*PULSE_DestroyLayerFunctionPtr)(struct PULSE_Layer *);

typedef enum
{
	PULSE_NONE,
	PULSE_DENSE,
	PULSE_CONV,
	PULSE_MAXPOLL
} PULSE_LayerType;

typedef struct PULSE_Layer
{
	PULSE_LayerType type;
	PULSE_OptimizationType optimization_type;
	PULSE_DataType *inputs;
	PULSE_DataType *outputs;
	PULSE_DataType *errors;
	PULSE_FeedLayerFunctionPtr feed;
	PULSE_BackLayerFunctionPtr back;
	PULSE_FixLayerFunctionPtr fix;
	PULSE_DestroyLayerFunctionPtr destroy;
	PULSE_ActivationFunctionPtr activate;
	struct PULSE_Layer * parent;
	struct PULSE_Layer * child;
	PULSE_Void * layer;
	PULSE_N n_inputs;
	PULSE_N n_outputs;
} PULSE_Layer;


PULSE_Layer PULSE_CreateLayer(PULSE_N , PULSE_N , PULSE_LayerType, PULSE_ActivationFunction, PULSE_FeedLayerFunctionPtr, PULSE_BackLayerFunctionPtr, PULSE_FixLayerFunctionPtr, PULSE_DestroyLayerFunctionPtr, PULSE_OptimizationType);
PULSE_Void PULSE_DestroyLayer();

#endif
