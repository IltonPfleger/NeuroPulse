#ifndef NETWORK_H
#define NETWORK_H


#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include "Layers.h"
#include "Activations.h"

void NN_Shuffle(int *indexes, int max)
{
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

void NN_Train(NN_Layer * input, NN_Layer * output, double * x, double * y, int data_size, int batch_size, int epoch, double lr)
{
	int random[data_size];
	for (int i = 0; i < epoch; i++)
	{
		NN_Shuffle(random, data_size);
		for (int j = 0; j < data_size; j++)
		{
			NN_FeedLayer(input, x + random[j] * input->n_inputs);
			for(int k = 0; k < output->size; k++)
				output->error[k] = 2*(output->outputs[k] - *((y + random[j] * output->size) + k));
			NN_BackLayer(output);

			if(j != 0 && j%batch_size == 0)
			{
				NN_Layer * current = input;
				while(current != NULL)
				{
					for(int i = 0; i < current->size; i++)
					{
						current->baias[i] -= lr * current->deltas[i]/batch_size;
						for(int j = 0; j < current->n_inputs; j++)
						{
							current->weights[i][j] -= lr * current->gradients[i][j]/batch_size;
							current->gradients[i][j] = 0;
						}
						current->deltas[i] = 0;
					}
					current = current->child;
				}
			}
		}
	}
}

#endif
