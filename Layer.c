#include "Include/Layer.h"

PULSE_Layer PULSE_CreateLayer(PULSE_N n_inputs, PULSE_N n_outputs, PULSE_LayerType type, PULSE_ActivationFunction activation_function, PULSE_FeedLayerFunctionPtr feed, PULSE_BackLayerFunctionPtr back, PULSE_FixLayerFunctionPtr fix, PULSE_DestroyLayerFunctionPtr destroy, PULSE_OptimizationType optimization){
	PULSE_Layer layer;
	layer.n_inputs = n_inputs;
	layer.n_outputs = n_outputs;
	layer.inputs = (PULSE_DataType*)PULSE_Alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType), n_inputs, optimization);
	layer.outputs = (PULSE_DataType*)PULSE_Alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType), n_outputs, optimization);
	layer.errors = (PULSE_DataType*)PULSE_Alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType), n_outputs, optimization);
	layer.type = type;
	layer.optimization_type = optimization;
	layer.child = NULL;
	layer.parent = NULL;
	layer.activate = PULSE_GetActivationFunctionPtr(activation_function);
	layer.feed = feed;
	layer.back = back;
	layer.fix = fix;
	layer.destroy = destroy;
	return layer;
}

void PULSE_DestroyLayer(PULSE_Layer * this)
{
	PULSE_Free(this->inputs);
	PULSE_Free(this->outputs);
	PULSE_Free(this->errors);
}
