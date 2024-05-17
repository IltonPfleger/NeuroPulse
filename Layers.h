#ifndef LAYERS_H
#define LAYERS_H

struct NN_LayerStruct;
struct NN_ConvolutionalLayerStruct;

typedef void (*NN_ActivationFunctionPtr)(struct NN_LayerStruct *, int);
typedef void (*NN_LayerFeedFunctionPtr)(struct NN_LayerStruct *, double *);

typedef struct NN_LayerStruct
{
	int size;
	int n_inputs;
	double **weights;
	double *baias;
	double *inputs;
	double *outputs;
	double *deltas;
	double **gradients;
	double *error;
	struct NN_LayerStruct *parent;
	struct NN_LayerStruct *child;
	NN_ActivationFunctionPtr activate;
	NN_LayerFeedFunctionPtr feed;
} NN_Layer;


void NN_FeedLayer(NN_Layer * layer, double * inputs)
{
	if(inputs != NULL)
		for(int i = 0; i < layer->n_inputs; i++)
			layer->inputs[i] = inputs[i];

	for(int i = 0; i < layer->size; i++)
	{
		layer->outputs[i] = 0;
		for(int j = 0; j < layer->n_inputs; j++)
			layer->outputs[i] += layer->inputs[j] * layer->weights[i][j];
		layer->outputs[i] += layer->baias[i];
	}
	layer->activate(layer, 0);
	if(layer->child != NULL)
	{
		for(int i = 0; i < layer->size; i++)
			layer->child->inputs[i] = layer->outputs[i];
		layer->child->feed(layer->child, NULL);
	}
}

void NN_BackLayer(NN_Layer * layer)
{
	if(layer->parent != NULL)
		for(int i = 0; i < layer->n_inputs; i++)
			layer->parent->error[i] = 0;

	layer->activate(layer, 1);
	for(int i = 0; i < layer->size; i++)
	{
		double delta = layer->error[i] * layer->outputs[i];
		layer->deltas[i] += delta;
		for(int j = 0; j < layer->n_inputs; j++)
		{
			layer->gradients[i][j] += delta * layer->inputs[j];
			if(layer->parent != NULL)
				layer->parent->error[j] += layer->weights[i][j] * delta;
		}
	}
	if(layer->parent != NULL)
		NN_BackLayer(layer->parent);
}

NN_Layer NN_CreateLayer(int n_inputs, int size, NN_ActivationFunctionPtr activate)
{
	NN_Layer layer;
	layer.size = size;
	layer.n_inputs = n_inputs;
	layer.parent = NULL;
	layer.child = NULL;
	layer.weights = (double **)malloc(sizeof(double *) * layer.size);
	layer.baias = (double *)malloc(sizeof(double) * layer.size);
	layer.inputs = (double *)malloc(sizeof(double) * layer.n_inputs);
	layer.outputs = (double *)malloc(sizeof(double) * layer.size);
	layer.deltas = (double *)malloc(sizeof(double) * layer.size);
	layer.gradients = (double **)malloc(sizeof(double *) * layer.size);
	layer.error = (double *)malloc(sizeof(double) * layer.size);
	layer.feed = &NN_FeedLayer;
	layer.activate = activate;
	for (int i = 0; i < layer.size; i++)
	{
		layer.weights[i] = (double *)malloc(sizeof(double) * layer.n_inputs);
		layer.gradients[i] = (double *)malloc(sizeof(double) * layer.n_inputs);
		layer.baias[i] = 0;
		layer.error[i] = 0;
		layer.outputs[i] = 0;
		layer.deltas[i] = 0;
		for (int j = 0; j < layer.n_inputs; j++)
		{
			layer.weights[i][j] = ((double)rand()/(double)(RAND_MAX))*sqrt(2.0/(double)(size + n_inputs));
			layer.gradients[i][j] = 0;
		}
	}
	return layer;
}

void NN_ConnectLayers(NN_Layer * layer, NN_Layer * next)
{
	layer->child = next;
	next->parent = layer;
}

#endif
