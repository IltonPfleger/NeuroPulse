#ifndef _PULSE_H_
#define _PULSE_H_
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "PULSE.h"

void PULSE_Shuffle(int *indexes, int max)
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

void PULSE_Train(PULSE_Layer * input, double * x, double * y, int data_size, int batch_size, int epoch, double lr)
{
	PULSE_Layer * output = input;
	while(output->child != NULL)
		output = output->child;

	int random[data_size];
	for (int i = 0; i < epoch; i++)
	{
		PULSE_Shuffle(random, data_size);
		for (int j = 0; j < data_size; j++)
		{
			input->feed(input, x + random[j] * input->n);
			for(int k = 0; k < output->size; k++)
				output->error[k] = 2*(output->outputs[k] - *((y + random[j] * output->size) + k));
			output->back(output);

			if(j != 0 && j%batch_size == 0)
			{
				PULSE_Layer * current = input;
				while(current != NULL)
				{
					current->fix(current, batch_size, lr);
					current = current->child;
				}
			}
		}
	}
}

void PULSE_Connect(PULSE_Layer * parent, PULSE_Layer * child)
{
	parent->child = child;
	child->parent = parent;
}

#endif
