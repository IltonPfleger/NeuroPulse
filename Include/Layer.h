#ifndef _PULSE_LAYER
#define _PULSE_LAYER
#include "PulseTypes.h"
#include "Activations.h"

struct PULSE_Layer;
typedef PULSE_Void (*PULSE_FeedLayerFunctionPtr)(struct PULSE_Layer *);
typedef PULSE_Void (*PULSE_BackLayerFunctionPtr)(struct PULSE_Layer *);
typedef PULSE_Void (*PULSE_FixLayerFunctionPtr)(struct PULSE_Layer *, PULSE_HyperArgs);
typedef PULSE_Void (*PULSE_DestroyLayerFunctionPtr)(struct PULSE_Layer *);

typedef enum
{
	PULSE_DENSE,
} PULSE_LayerType;

typedef struct {
	PULSE_DataType * weights;
	PULSE_DataType * baiases;
	PULSE_DataType * deltas;
	PULSE_DataType * ddeltas;
	PULSE_DataType * gradients;
}PULSE_DenseLayer;

typedef union
{
	PULSE_DenseLayer DENSE;
} PULSE_LayerWrapper;



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
	PULSE_LayerWrapper layer;
	PULSE_N n_inputs;
	PULSE_N n_outputs;
} PULSE_Layer;


PULSE_Layer PULSE_CreateLayer(PULSE_N , PULSE_N , PULSE_LayerType, PULSE_ActivationFunction, PULSE_FeedLayerFunctionPtr, PULSE_BackLayerFunctionPtr, PULSE_FixLayerFunctionPtr, PULSE_DestroyLayerFunctionPtr, PULSE_OptimizationType);
PULSE_Void PULSE_DestroyLayer(PULSE_Layer*);

#endif
