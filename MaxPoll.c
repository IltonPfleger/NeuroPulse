#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "Include/MaxPoll.h"


static void _FeedMaxPoll(PULSE_Layer * this)
{
	PULSE_MaxPollLayer * poll = (PULSE_MaxPollLayer*)this->layer;
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



static void _BackMaxPoll(PULSE_Layer * this)
{
	if(this->parent != NULL)
	{
		PULSE_MaxPollLayer * poll = (PULSE_MaxPollLayer*)this->layer;
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



static void _FixMaxPoll(PULSE_Layer * this, PULSE_HyperArgs args){}

static void _DestroyMaxPoll(PULSE_Layer * this)
{
	free(this->layer);
	PULSE_DestroyLayer(this);
}


PULSE_Layer PULSE_CreateMaxPollLayer(int k_size, int iz, int iy, int ix)
{
	int n_inputs = iz*iy*ix;
	int n_outputs = iz*(iy/k_size)*(ix/k_size);
	PULSE_MaxPollLayer * poll = (PULSE_MaxPollLayer*)malloc(sizeof(PULSE_MaxPollLayer));
	poll->k_size = k_size;
	poll->i_size[0] = iz;
	poll->i_size[1] = iy;
	poll->i_size[2] = ix;
	poll->o_size[0] = iz;
	poll->o_size[1] = iy/k_size;
	poll->o_size[2] = ix/k_size;
	PULSE_Layer layer = PULSE_CreateLayer(n_inputs, n_outputs, PULSE_MAXPOLL, &_FeedMaxPoll, &_BackMaxPoll, &_FixMaxPoll, &_DestroyMaxPoll);
	layer.layer = poll;
	return layer;
}


