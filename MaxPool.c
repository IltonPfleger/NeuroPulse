#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "Include/MaxPool.h"


static void _FeedMaxPool(PULSE_Layer * this)
{
	PULSE_MaxPoolLayer * poll = (PULSE_MaxPoolLayer*)this->layer;
#pragma omp parallel for schedule(static)
	for(int i = 0; i < poll->i_size[0]; i++)
	{
		int i_z = i * poll->i_size[1] * poll->i_size[2];
		int o_z = i * poll->o_size[1] * poll->o_size[2];
		for(int j = 0; j + poll->k_size <= poll->i_size[1]; j += poll->k_size)
		{
			int o_y = j/poll->k_size * poll->o_size[2];
			for(int k = 0; k + poll->k_size <= poll->i_size[2]; k += poll->k_size)
			{
				int o_x = k/poll->k_size;
				PULSE_DataType max = -INFINITY;
				for(int m = 0; m < poll->k_size; m++)
					for(int n = 0; n < poll->k_size; n++)
					{
						int input_index = i_z + (j + m) * poll->i_size[2] + (k + n);
						if(max < this->inputs[input_index])
							max = this->inputs[input_index];
					}
				this->outputs[o_z + o_y + o_x] = max;
			}
		}
	}
}



static void _BackMaxPool(PULSE_Layer * this)
{
	if(this->parent != NULL)
	{
		PULSE_MaxPoolLayer * poll = (PULSE_MaxPoolLayer*)this->layer;
#pragma omp parallel for schedule(static)
		for(int i = 0; i < poll->i_size[0]; i++)
		{
			int i_z = i * poll->i_size[1] * poll->i_size[2];
			int o_z = i * poll->o_size[1] * poll->o_size[2];
			for(int j = 0; j + poll->k_size <= poll->i_size[1]; j += poll->k_size)
			{
				int o_y = j/poll->k_size * poll->o_size[2];
				for(int k = 0; k + poll->k_size <= poll->i_size[2]; k += poll->k_size)
				{
					int o_x = k/poll->k_size;
					int output_index = o_z + o_y + o_x;
					for(int m = 0; m < poll->k_size; m++)
						for(int n = 0; n < poll->k_size; n++)
						{
							int input_index = i_z + (j + m) * poll->i_size[2] + (k + n);
							if(this->inputs[input_index] == this->outputs[output_index])
								this->parent->errors[input_index] = this->errors[output_index];
							else 
								this->parent->errors[input_index] = 0;
						}
				}
			}
		}
	}
}



static void _FixMaxPool(PULSE_Layer * this, PULSE_HyperArgs args){}

static void _DestroyMaxPool(PULSE_Layer * this)
{
	free(this->layer);
	PULSE_DestroyLayer(this);
}


PULSE_Layer PULSE_CreateMaxPoolLayer(PULSE_N k_size, PULSE_N iz, PULSE_N iy, PULSE_N ix)
{
	int n_inputs = iz*iy*ix;
	int n_outputs = iz*(iy/k_size)*(ix/k_size);
	PULSE_MaxPoolLayer * poll = (PULSE_MaxPoolLayer*)malloc(sizeof(PULSE_MaxPoolLayer));
	poll->k_size = k_size;
	poll->i_size[0] = iz;
	poll->i_size[1] = iy;
	poll->i_size[2] = ix;
	poll->o_size[0] = iz;
	poll->o_size[1] = iy/k_size;
	poll->o_size[2] = ix/k_size;
	PULSE_Layer layer = PULSE_CreateLayer(n_inputs, n_outputs, PULSE_MAXPOLL, &_FeedMaxPool, &_BackMaxPool, &_FixMaxPool, &_DestroyMaxPool);
	layer.layer = poll;
	return layer;
}


