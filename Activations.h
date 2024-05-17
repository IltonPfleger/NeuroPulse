#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H




void NN_Sigmoid(NN_Layer * layer, int prime)
{
	for(int i = 0; i < layer->size; i++)
	{
		double *x = &layer->outputs[i];
		if(prime)
			*x = *x*(1.f - *x);
		*x = 1.f/(1.f + expf(-*x));
	}
}


void NN_Softmax(NN_Layer * layer, int prime)
{


	if(prime)
	{
		for(int i = 0; i < layer->size; i++)
			layer->outputs[i] = layer->outputs[i]*(1 - layer->outputs[i]);
		return;
	}

	double max = layer->outputs[0];
	for (int i = 1; i < layer->size; i++)
		if (layer->outputs[i] > max) 
			max = layer->outputs[i];


	double sum = 0;
	for(int i = 0; i < layer->size; i++)
	{
		layer->outputs[i] = exp(layer->outputs[i] - max);
		sum += layer->outputs[i];
	}

	for(int i = 0; i < layer->size; i++)
		layer->outputs[i] /= sum;
}


void NN_ReLU(NN_Layer * layer, int prime)
{
	for(int i = 0; i < layer->size; i++)
	{
		double* x = &layer->outputs[i];
		if(prime)
			*x = *x > 0 ? 1 : 0.0001;
		*x = *x > 0 ? *x: *x*0.0001;
	}
}

#endif
