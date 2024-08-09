#include "Include/Dense.h"
#include "Include/PULSE_SIMD.h"



static void _FeedDense(PULSE_Layer * this)
{
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
	{
		this->outputs[i] = 0;
		for(int j = 0; j < this->n_inputs; j++)
			this->outputs[i] += this->inputs[j] * dense->weights[wi + j];
		this->outputs[i] += dense->baiases[i];
	}
	this->activate(this, 0);
}

static void _SIMD_FeedDense(PULSE_Layer * this)
{
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	__PULSE_SIMD_1MNxMMN_ADD(this->inputs, dense->weights, dense->baiases, this->outputs, this->n_outputs, this->n_inputs);
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


static void _SIMD_BackDense(PULSE_Layer * this)
{
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	this->activate(this, 1);
	__PULSE_SIMD_MUL_1x1(this->errors, this->outputs, dense->ddeltas, this->n_outputs);
	__PULSE_SIMD_ADD_1x1(dense->ddeltas, dense->deltas, dense->deltas, this->n_outputs);
	__PULSE_SIMD_BIGEST_DATA_TYPE delta, gradients, inputs, weights, errors;

	for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
	{
		delta = __PULSE_SIMD_SET_ALL(dense->ddeltas[i]);
		for(int j = 0; j < this->n_inputs; j += __PULSE_SIMD_N_PER_CHUNK)
		{
			gradients = __PULSE_SIMD_LOAD(dense->gradients + wi + j);
			inputs = __PULSE_SIMD_LOAD(this->inputs + j);
			gradients = __PULSE_SIMD_MADD(delta, inputs, gradients);
			__PULSE_SIMD_GET(&dense->gradients[wi + j], gradients);

			if(this->parent != NULL)
			{
				weights = __PULSE_SIMD_LOAD(dense->weights + wi + j);
				errors = __PULSE_SIMD_LOAD(this->parent->errors + j);
				__PULSE_SIMD_GET(this->parent->errors + j ,__PULSE_SIMD_MADD(weights, delta, errors));
			}
		}
	}
}




static void _FixDense(PULSE_Layer * this, PULSE_HyperArgs args)
{
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	const double HYPER = args.lr/args.batch_size;
	for (int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
	{
		dense->baiases[i] -= HYPER * dense->deltas[i];
		dense->deltas[i] = 0;
		for (int j = 0; j < this->n_inputs; j++)
		{
			dense->weights[wi + j] -= HYPER * dense->gradients[wi + j];
			dense->gradients[wi + j] = 0;
		}
	}
}

static void _SIMD_FixDense(PULSE_Layer * this, PULSE_HyperArgs args)
{
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	__PULSE_SIMD_BIGEST_DATA_TYPE baiases, deltas, weights, gradients;
	const __PULSE_SIMD_BIGEST_DATA_TYPE HYPER = __PULSE_SIMD_SET_ALL(-args.lr/args.batch_size);
	const __PULSE_SIMD_BIGEST_DATA_TYPE ZERO = __PULSE_SIMD_ZERO();

	for(int i = 0; i < this->n_outputs; i += __PULSE_SIMD_N_PER_CHUNK)
	{
		baiases = __PULSE_SIMD_LOAD(dense->baiases + i);
		deltas = __PULSE_SIMD_LOAD(dense->deltas + i);
		baiases = __PULSE_SIMD_ADD(baiases, __PULSE_SIMD_MUL(deltas, HYPER));
		__PULSE_SIMD_GET(dense->baiases + i, baiases);
		__PULSE_SIMD_GET(dense->deltas + i, ZERO);
	}

	for(int i = 0; i < this->n_outputs * this->n_inputs; i += __PULSE_SIMD_N_PER_CHUNK)
	{
		weights = __PULSE_SIMD_LOAD(dense->weights + i);
		gradients = __PULSE_SIMD_LOAD(dense->gradients + i);
		weights = __PULSE_SIMD_ADD(weights, __PULSE_SIMD_MUL(gradients, HYPER));
		__PULSE_SIMD_GET(dense->weights + i, weights);
		__PULSE_SIMD_GET(dense->gradients + i, ZERO);
	}
}




static void _DestroyDense(PULSE_Layer * this)
{
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	free(dense->weights);
	free(dense->baiases);
	free(dense->gradients);
	free(dense->deltas);
	free(dense);
	PULSE_DestroyLayer(this);
}



PULSE_Layer PULSE_CreateDenseLayer(int n_inputs, int n_outputs, PULSE_ActivationLayerFunctionPtr activation_function, PULSE_OptimizationType optimization)
{
	PULSE_DenseLayer *dense = (PULSE_DenseLayer*)malloc(sizeof(PULSE_DenseLayer));

	dense->weights = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*n_inputs*n_outputs);
	dense->gradients = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*n_inputs*n_outputs);
	dense->baiases = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*n_outputs);
	dense->deltas = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*n_outputs);
	dense->ddeltas = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*n_outputs);

	for(int i = 0; i < n_inputs*n_outputs; i++)
		dense->weights[i] = (PULSE_DataType)rand()/(PULSE_DataType)(RAND_MAX)*sqrt(2.0/(PULSE_DataType)(n_inputs+n_outputs));

	PULSE_Layer layer;
	switch(optimization)
	{
		case PULSE_OPTIMIZATION_NONE:
			layer = PULSE_CreateLayer(n_inputs, n_outputs, PULSE_DENSE, activation_function, &_FeedDense, &_BackDense, &_FixDense, &_DestroyDense, optimization);
			break;
		case PULSE_OPTIMIZATION_SIMD:
			layer = PULSE_CreateLayer(n_inputs, n_outputs, PULSE_DENSE, activation_function, &_SIMD_FeedDense, &_SIMD_BackDense, &_SIMD_FixDense, &_DestroyDense, optimization);
			break;
	}

	layer.layer = dense;
	return layer;
}


