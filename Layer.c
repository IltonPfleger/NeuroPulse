#include "Include/Layer.h"

PULSE_Layer PULSE_CreateLayer(PULSE_N n_inputs, PULSE_N n_outputs, PULSE_LayerType type, PULSE_ActivationLayerFunctionPtr activate, PULSE_FeedLayerFunctionPtr feed, PULSE_BackLayerFunctionPtr back, PULSE_FixLayerFunctionPtr fix, PULSE_DestroyLayerFunctionPtr destroy, PULSE_OptimizationType optimization_type){
	PULSE_Layer layer;
	layer.n_inputs = n_inputs;
	layer.n_outputs = n_outputs;
	layer.inputs = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*n_inputs);
	layer.outputs = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*n_outputs);
	layer.errors = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*n_outputs);
	layer.type = type;
	layer.optimization_type = optimization_type;
	layer.child = NULL;
	layer.parent = NULL;
	layer.activate = activate;
	layer.feed = feed;
	layer.back = back;
	layer.fix = fix;
	layer.destroy = destroy;
	return layer;
}

void PULSE_DestroyLayer(PULSE_Layer * this)
{
	free(this->inputs);
	free(this->outputs);
	free(this->errors);
}
