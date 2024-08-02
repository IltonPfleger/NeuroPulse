#include "Include/Convolutional.h"

static void _FeedConvolutional(PULSE_Layer * this)
{
	PULSE_ConvolutionalLayer conv = *(PULSE_ConvolutionalLayer*)this->layer;
	const int iz_size = conv.i_size[1]*conv.i_size[2];
	const int oz_size = conv.o_size[1]*conv.o_size[2];
	const int k2 = conv.k_size * conv.k_size;

#pragma omp parallel for schedule(static) collapse(2)
	for(int i = 0; i < conv.o_size[0]; i++)
	{
		int o_z = i * oz_size;
		int k_i = i * conv.i_size[0] * k2;
		int k_z = 0;
		int i_z = 0;
		for(int j = 0; j < conv.i_size[0]; j++)
		{
			int o_y = 0;
			for(int k = 0, ks2 = 0; k + conv.k_size <= conv.i_size[1]; k++, o_y += conv.o_size[2], ks2 += conv.i_size[2])
				for(int l = 0; l + conv.k_size <= conv.i_size[2]; l++)
				{
					PULSE_DataType sum = 0;
					for(int m = 0, m_k_size = 0, ms2 = 0; m < conv.k_size; m++, m_k_size += conv.k_size, ms2 += conv.i_size[2])
						for(int n = 0; n < conv.k_size; n++)
							sum += this->inputs[i_z + (ks2 + ms2) + (l + n)] * conv.kernels[k_i + k_z + m_k_size + n];
					this->outputs[o_z + o_y + l] = sum + conv.baias[o_z + o_y + l];
				}
			k_z += k2;
			i_z += iz_size;
		}
	}
}

static void _BackConvolutional(PULSE_Layer * this)
{
	PULSE_ConvolutionalLayer conv = *(PULSE_ConvolutionalLayer*)this->layer;
	const int iz_size = conv.i_size[1]*conv.i_size[2];
	const int oz_size = conv.o_size[1]*conv.o_size[2];
	const int k2 = conv.k_size * conv.k_size;


#pragma omp parallel for schedule(static) collapse(2)
	for(int i = 0; i < conv.o_size[0]; i++)
	{
		int k_i = i * conv.i_size[0] * k2;
		int error_z = i * iz_size;
		for(int j = 0; j < conv.i_size[0]; j++)
		{
			for(int k = -conv.k_size, k_z = 0, parent_y = 0; k + conv.o_size[1] <= conv.i_size[1]; k++, k_z += k2, parent_y++)
			{
				for(int l = -conv.k_size, parent_x = 0; l + conv.o_size[2] <= conv.i_size[2]; l++, parent_x++)
				{
					PULSE_DataType sum = 0;
					for(int m = 0; m < conv.o_size[1]; m++)
					{
						for(int n = 0; n < conv.o_size[2]; n++)
						{
							//Local Gradient
							if(k > 0 && l > 0)
							{
								int input_index = (j * conv.i_size[1] * conv.i_size[2]) + ((k + m) * conv.i_size[2]) + l + n;
								int error_index = (i * conv.o_size[1] * conv.o_size[2]) + (m * conv.o_size[2]) + n;
								conv.gradients[k_i + k_z + m*conv.k_size + n] += this->errors[error_index] * this->inputs[input_index];
								if(j == 0 && k == 0 && l == 0)
									conv.deltas[error_index] += this->errors[error_index];
							}

							//Previous Error
							if(this->parent != NULL)
								if(m < conv.k_size && n < conv.k_size)
								{
									int error_y = (k+m) * conv.o_size[2];
									int error_x = l + n;
									if(error_y > 0 && error_y < conv.o_size[1] && error_x > 0 && error_x < conv.o_size[2])
										this->parent->errors[j + parent_y + parent_x] = conv.kernels[k_i + k_z + (conv.k_size-m)*conv.k_size + (conv.k_size - n)] * this->errors[error_z + error_y + error_y];
								}
						}
					}
				}
			}
		}
	}
}



static void _FixConvolutional(PULSE_Layer * this, PULSE_HyperArgs args)
{
	PULSE_ConvolutionalLayer conv = *(PULSE_ConvolutionalLayer*)this->layer;
	const int iz_size = conv.i_size[1]*conv.i_size[2];
	const int oz_size = conv.o_size[1]*conv.o_size[2];
	const int k2 = conv.k_size * conv.k_size;


#pragma omp parallel for schedule(static) collapse(2)
	for(int i = 0; i < conv.o_size[0]; i++)
	{
		int k_i = i * conv.i_size[0] * k2;
		int baias_z = i * oz_size;
		for(int j = 0; j < conv.i_size[0]; j++)
			for(int k = 0, k_z = 0, baias_y = 0; k < conv.k_size; k++, k_z += k2, baias_y += conv.k_size)
				for(int l = 0; l < conv.k_size; l++)
				{
					int k_index = k_i + k_z + (k * conv.k_size) + l;
					conv.kernels[k_index] -= args.lr * conv.gradients[k_index] / args.batch_size;
					conv.gradients[k_index] = 0;
					if(j == 0)
					{
						conv.baias[baias_z + baias_y + l] -= args.lr * conv.deltas[baias_z + baias_y + l] / args.batch_size;
						conv.deltas[baias_z + baias_y + l] = 0;
					}
				}
	}
}


static void _DestroyConvolutional(PULSE_Layer * this)
{

	PULSE_ConvolutionalLayer * conv = (PULSE_ConvolutionalLayer*)this->layer;
	free(conv->kernels);
	free(conv->gradients);
	free(conv->baias);
	free(conv->deltas);
	free(conv);
	PULSE_DestroyLayer(this);
}


PULSE_Layer PULSE_CreateConvolutionalLayer(int dim, int k_size, int iz, int iy, int ix)
{
	int n_inputs = iz*iy*ix;
	int n_outputs = dim * (iy - k_size + 1) * (ix - k_size + 1);
	int o_xy_size = (iy - k_size + 1) * (ix - k_size + 1);
	int k2 = k_size*k_size;
	PULSE_ConvolutionalLayer *conv = (PULSE_ConvolutionalLayer*)malloc(sizeof(PULSE_ConvolutionalLayer));
	conv->i_size[0] = iz;
	conv->i_size[1] = iy;
	conv->i_size[2] = ix;
	conv->o_size[0] = dim;
	conv->o_size[1] = iy - k_size + 1;
	conv->o_size[2] = ix - k_size + 1;
	conv->k_size = k_size;
	conv->kernels = (PULSE_DataType*)malloc(sizeof(PULSE_DataType)*dim*ix*k_size*k_size);
	conv->gradients = (PULSE_DataType*)calloc(dim*ix*k_size*k_size, sizeof(PULSE_DataType));
	conv->baias = (PULSE_DataType*)calloc(n_outputs, sizeof(PULSE_DataType));
	conv->deltas = (PULSE_DataType*)calloc(n_outputs, sizeof(PULSE_DataType));

	for(int i = 0; i < dim*iz*k_size*k_size; i++)
		conv->kernels[i] = (PULSE_DataType)rand()/(PULSE_DataType)(RAND_MAX);

	PULSE_Layer layer = PULSE_CreateLayer(n_inputs, n_outputs, PULSE_CONV, &_FeedConvolutional, &_BackConvolutional, &_FixConvolutional, &_DestroyConvolutional);
	layer.layer = conv;
	return layer;
}
