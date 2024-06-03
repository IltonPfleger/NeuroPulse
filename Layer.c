#include <stdio.h>
#include <stdlib.h>
#include "Include/Layer.h"

PULSE_Layer PULSE_CreateLayer(int n_inputs, int n_outputs, PULSE_LayerType type, PULSE_FeedLayerFunctionPtr feed, PULSE_BackLayerFunctionPtr back, PULSE_FixLayerFunctionPtr fix, PULSE_DestroyLayerFunctionPtr destroy){
	PULSE_Layer layer;
	layer.n_inputs = n_inputs;
	layer.n_outputs = n_outputs;
	layer.inputs = (PULSE_DataType*)calloc(n_inputs, sizeof(PULSE_DataType));
	layer.outputs = (PULSE_DataType*)calloc(n_outputs, sizeof(PULSE_DataType));
	layer.errors = (PULSE_DataType*)calloc(n_outputs, sizeof(PULSE_DataType));
	layer.type = type;
	layer.child = NULL;
	layer.parent = NULL;
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
