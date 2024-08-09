#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "Include/Layer.h"
#include "Include/Dense.h"
#include "Include/MaxPool.h"
#include "Include/Convolutional.h"
#include "Include/Activations.h"
#include "Include/PULSE.h"


PULSE_DataType * PULSE_Foward(PULSE_Layer * layer, PULSE_DataType * inputs)
{
	if(inputs != NULL)
		memcpy(layer->inputs, inputs, sizeof(PULSE_DataType)*layer->n_inputs);
	layer->feed(layer);

	if(layer->child != NULL)
	{
		memcpy(layer->child->inputs, layer->outputs, sizeof(PULSE_DataType)*layer->n_outputs);
		return PULSE_Foward(layer->child, NULL);
	}
	else
		return layer->outputs;
}


void PULSE_Back(PULSE_Layer * layer)
{
	layer->back(layer);
	if(layer->parent != NULL)
	{
		PULSE_Back(layer->parent);
		memset(layer->parent->errors, 0, layer->n_inputs*sizeof(PULSE_DataType));
	}
}



void PULSE_Shuffle(PULSE_N *indexes, PULSE_N max)
{
	srand(time(NULL));
	for (int i = 0; i < max; i++)
		indexes[i] = i;
	for (int i = 0; i < max; i++)
	{
		int random = (int)rand() % max;
		int random2 = (int)rand() % max;
		int temp = indexes[random];
		indexes[random] = indexes[random2];
		indexes[random2] = temp;
	};
}



void PULSE_Train(PULSE_Layer * first_layer, PULSE_N epoch, PULSE_N data_size, PULSE_HyperArgs args, PULSE_DataType * x, PULSE_DataType * y)
{
	PULSE_Layer * output = first_layer;
	while(output->child != NULL)
		output = output->child;

	int random[data_size];
	for (int i = 0; i < epoch; i++)
	{
		PULSE_Shuffle(random, data_size);
		for (int j = 0; j < data_size; j++)
		{
			PULSE_Foward(first_layer, x + random[j] * first_layer->n_inputs);
			for(int k = 0; k < output->n_outputs; k++)
				output->errors[k] = 2*(output->outputs[k] - *((y + random[j] * output->n_outputs) + k));

			PULSE_Back(output);

			if((j+1)%args.batch_size == 0)
			{
				PULSE_Layer * current = first_layer;
				while(current != NULL)
				{
					current->fix(current, args);
					current = current->child;
				}

			}
			printf("Epoch: %d Item: %d\r", i, j);
		}
	}
}

void PULSE_Connect(PULSE_Layer * parent, PULSE_Layer * child)
{
	parent->child = child;
	child->parent = parent;
}

void PULSE_Destroy(PULSE_Layer * layer)
{
	PULSE_Layer * current = layer;
	while(current)
	{
		current->destroy(current);
		current = current->child;
	}
}
