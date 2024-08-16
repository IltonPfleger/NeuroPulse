#ifndef _PULSE_LAYER
#define _PULSE_LAYER
#include "PulseTypes.h"
#include "Activations.h"

struct PULSE_Layer;
typedef PULSE_Void (*PULSE_FeedLayerFunctionPtr)(struct PULSE_Layer *);
typedef PULSE_Void (*PULSE_BackLayerFunctionPtr)(struct PULSE_Layer *);
typedef PULSE_Void (*PULSE_FixLayerFunctionPtr)(struct PULSE_Layer *, PULSE_HyperArgs);
typedef PULSE_Size_t (*PULSE_DistributeTrainLayerAllocation)(struct PULSE_Layer *, PULSE_DataType *);

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
	PULSE_DataType *inputs;
	PULSE_DataType *outputs;
	PULSE_DataType *errors;
	PULSE_LayerType type;
	PULSE_OptimizationType optimization;
	PULSE_FeedLayerFunctionPtr feed;
	PULSE_BackLayerFunctionPtr back;
	PULSE_FixLayerFunctionPtr fix;
	PULSE_ActivationFunctionPtr activate;
	PULSE_DistributeTrainLayerAllocation pull_tmp;
	struct PULSE_Layer * parent;
	struct PULSE_Layer * child;
	PULSE_LayerWrapper layer;
	unsigned int n_inputs;
	unsigned int n_outputs;
} PULSE_Layer;


#endif
