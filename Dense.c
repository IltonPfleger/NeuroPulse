#include "Include/Dense.h"



static void _FeedDense(PULSE_Layer * this)
{
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
	{
		this->outputs[i] = 0;
		for(int j = 0; j < this->n_inputs; j++)
			this->outputs[i] += this->inputs[j] * dense->weights[wi + j];
		this->outputs[i] += dense->baias[i];
	}
	this->activate(this, 0);
}



static void _BackDense(PULSE_Layer * this)
{
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	this->activate(this, 1);
	for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
	{
		double delta = this->errors[i] * this->outputs[i];
		dense->deltas[i] += delta;
		for(int j = 0; j < this->n_inputs; j++)
		{
			dense->gradients[wi + j] += delta * this->inputs[j];
			if(this->parent != NULL)
				this->parent->errors[j] += dense->weights[wi + j] * delta;
		}
	}
}


static void _FixDense(PULSE_Layer * this, PULSE_HyperArgs args)
{
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;

	for (int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
	{
		dense->baias[i] -= args.lr * dense->deltas[i] / args.batch_size;
		dense->deltas[i] = 0;
		for (int j = 0; j < this->n_inputs; j++)
		{
			dense->weights[wi + j] -= args.lr * dense->gradients[wi + j] / args.batch_size;
			dense->gradients[wi + j] = 0;
		}
	}
}


static void _DestroyDense(PULSE_Layer * this)
{
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	free(dense->weights);
	free(dense->baias);
	free(dense->gradients);
	free(dense->deltas);
	free(dense);
	PULSE_DestroyLayer(this);
}



PULSE_Layer PULSE_CreateDenseLayer(int n_inputs, int n_outputs, PULSE_ActivationLayerFunctionPtr activation_function)
{
	PULSE_DenseLayer *dense = (PULSE_DenseLayer*)malloc(sizeof(PULSE_DenseLayer));
	dense->weights = (PULSE_DataType*)malloc(sizeof(PULSE_DataType)*n_inputs*n_outputs);
	dense->gradients = (PULSE_DataType*)calloc(n_inputs*n_outputs, sizeof(PULSE_DataType));
	dense->baias = (PULSE_DataType*)calloc(n_outputs, sizeof(PULSE_DataType*));
	dense->deltas = (PULSE_DataType*)calloc(n_outputs, sizeof(PULSE_DataType*));

	for(int i = 0; i < n_inputs*n_outputs; i++)
		dense->weights[i] = (PULSE_DataType)rand()/(PULSE_DataType)(RAND_MAX)*sqrt(2.0/(PULSE_DataType)(n_inputs+n_outputs));

	PULSE_Layer layer = PULSE_CreateLayer(n_inputs, n_outputs, PULSE_DENSE, &_FeedDense, &_BackDense, &_FixDense, &_DestroyDense);
	layer.layer = dense;
	layer.activate = activation_function;
	return layer;
}


