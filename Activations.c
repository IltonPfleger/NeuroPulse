#include "Include/Activations.h"

void PULSE_Sigmoid(PULSE_Layer * layer, char prime)
{
	for(int i = 0; i < layer->n_outputs; i++)
	{
		if(prime)
			layer->outputs[i] = layer->outputs[i]*(1.f - layer->outputs[i]);
		layer->outputs[i] = 1.f/(1.f + expf(-layer->outputs[i]));
	}
}


void PULSE_ReLU(PULSE_Layer * layer, char prime)
{

	for(int i = 0; i < layer->n_outputs; i++)
	{
		if(prime)
			layer->outputs[i] = layer->outputs[i] > 0 ? 1:0.0001;
		layer->outputs[i] = layer->outputs[i] > 0 ? layer->outputs[i]:0.0001*layer->outputs[i];
	}
}



//void PULSE_Softmax(PULSE_Layer * layer, char prime)
//{
//	if(prime)
//	{
//		for(int i = 0; i < layer->n_outputs; i++)
//			layer->outputs[i] = layer->outputs[i]*(1.f - layer->outputs[i]);
//		return;
//	}
//
//	double max = layer->outputs[0];
//
//	for (int i = 1; i < layer->n_outputs; i++)
//		if (layer->outputs[i] > max) 
//			max = layer->outputs[i];
//
//
//	double sum = 0;
//
//	for(int i = 0; i < layer->n_outputs; i++)
//	{
//		layer->outputs[i] = exp(layer->outputs[i] - max);
//		sum += layer->outputs[i];
//	}
//
//	for(int i = 0; i < layer->n_outputs; i++)
//		layer->outputs[i] /= sum;
//}
