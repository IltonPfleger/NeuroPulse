#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "Layer.h"

static void PULSE_FowardLayer(PULSE_Layer * this, double * inputs)
{
	if(inputs != NULL)
		for(int i = 0; i < this->n; i++)
			this->inputs[i] = inputs[i];

	for(int i = 0; i < this->size; i++)
	{
		this->outputs[i] = 0;
		for(int j = 0; j < this->n; j++)
			this->outputs[i] += this->inputs[j] * this->weights[i * this->n + j];
		this->outputs[i] += this->baias[i];
	}
	this->activate(this->outputs, this->size, 0);
	if(this->child != NULL)
	{
		for(int i = 0; i < this->size; i++)
			this->child->inputs[i] = this->outputs[i];
		this->child->feed(this->child, NULL);
	}
}

static void PULSE_BackLayer(PULSE_Layer * this)
{
	if(this->parent != NULL)
		for(int i = 0; i < this->n; i++)
			this->parent->error[i] = 0;

	this->activate(this->outputs, this->size, 1);
	for(int i = 0; i < this->size; i++)
	{
		double delta = this->error[i] * this->outputs[i];
		this->deltas[i] += delta;
		for(int j = 0; j < this->n; j++)
		{
			this->gradients[i * this->n + j] += delta * this->inputs[j];
			if(this->parent != NULL)
				this->parent->error[j] += this->weights[i * this->n + j] * delta;
		}
	}
	if(this->parent != NULL)
		this->parent->back(this->parent);
}

static void PULSE_FixLayer(PULSE_Layer * this, int batch_size, double lr)
{
	for (int i = 0; i < this->size; i++)
	{
		this->baias[i] -= lr * this->deltas[i] / batch_size;
		this->deltas[i] = 0;
		for (int j = 0; j < this->n; j++)
		{
			this->weights[i * this->n + j] -= lr * this->gradients[i * this->n + j] / batch_size;
			this->gradients[i * this->n + j] = 0;
		}
	}
}


PULSE_Layer PULSE_CreateLayer(int n, int size, PULSE_ActivationFunctionPtr activate)
{
	PULSE_Layer this;
	this.n = n;
	this.size = size;
	this.parent = NULL;
	this.child = NULL;
	this.weights = (double *)malloc(sizeof(double) * this.size * this.n);
	this.gradients = (double *)malloc(sizeof(double) * this.size * this.n);
	this.baias = (double *)malloc(sizeof(double) * this.size);
	this.inputs = (double *)malloc(sizeof(double) * this.n);
	this.outputs = (double *)malloc(sizeof(double) * this.size);
	this.deltas = (double *)malloc(sizeof(double) * this.size);
	this.error = (double *)malloc(sizeof(double) * this.size);
	this.feed = &PULSE_FowardLayer;
	this.back = &PULSE_BackLayer;
	this.fix = &PULSE_FixLayer;
	this.activate = activate;

	for (int i = 0; i < this.size; i++)
		for (int j = 0; j < this.n; j++)
			this.weights[i * this.n + j] = ((double)rand()/(double)(RAND_MAX))*sqrt(2.0/(double)(size + n));

	return this;
}


