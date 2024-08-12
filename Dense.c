#include "Include/Dense.h"
#include "Include/PULSE_SIMD.h"


//STD
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
	this->activate(this->outputs, this->n_outputs, 0);
}


static void _BackDense(PULSE_Layer * this)
{
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	this->activate(this->outputs, this->n_outputs, 1);
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

#if defined(__PULSE_SIMD_SUPPORTED)
static void _SIMD_FeedDense(PULSE_Layer * this)
{
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	__PULSE_SIMD_DATATYPE inputs, weights, outputs;
	for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
	{
		outputs = __PULSE_SIMD_ZERO();
		for(int j = 0; j < this->n_inputs; j += __PULSE_SIMD_N_PER_CHUNK)
		{
			weights = __PULSE_SIMD_LOAD(dense->weights + wi + j);
			inputs = __PULSE_SIMD_LOAD(this->inputs + j);
			if(j + __PULSE_SIMD_N_PER_CHUNK > this->n_inputs)
			{
				weights = __PULSE_SIMD_ZERO_R(weights, __PULSE_SIMD_N_PER_CHUNK - this->n_inputs - j);
				inputs = __PULSE_SIMD_ZERO_R(inputs, __PULSE_SIMD_N_PER_CHUNK - this->n_inputs - j);
			}
			outputs = __PULSE_SIMD_MADD(weights, inputs, outputs);
		}
		this->outputs[i] = __PULSE_SIMD_TO_FLOAT(__PULSE_SIMD_X86_REDUCE_ADD_256(outputs)) + dense->baiases[i];
	}
	this->activate(this->outputs, this->n_outputs, 0);
}


static void _SIMD_BackDense(PULSE_Layer * this)
{
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	this->activate(this->outputs, this->n_outputs, 1);
	__PULSE_SIMD_DATATYPE deltas, ddeltas, delta, errors, gradients, inputs, weights, outputs;

	for(int i = 0; i < this->n_outputs; i += __PULSE_SIMD_N_PER_CHUNK)
	{
		errors = __PULSE_SIMD_LOAD(this->errors + i);
		outputs = __PULSE_SIMD_LOAD(this->outputs + i);
		ddeltas = __PULSE_SIMD_LOAD(dense->deltas + i);
		deltas = __PULSE_SIMD_LOAD(dense->ddeltas + i);
		ddeltas = __PULSE_SIMD_MUL(errors, outputs);
		deltas = __PULSE_SIMD_ADD(ddeltas, deltas);
		__PULSE_SIMD_STORE(dense->ddeltas + i, ddeltas);
		__PULSE_SIMD_STORE(dense->deltas + i, deltas);
	};

	if(this->parent != NULL)
	{
		for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
		{
			delta = __PULSE_SIMD_SET_ALL(dense->ddeltas[i]);
			for(int j = 0; j < this->n_inputs; j += __PULSE_SIMD_N_PER_CHUNK)
			{
				gradients = __PULSE_SIMD_LOAD(dense->gradients + wi + j);
				inputs = __PULSE_SIMD_LOAD(this->inputs + j);
				gradients = __PULSE_SIMD_MADD(delta, inputs, gradients);
				__PULSE_SIMD_STORE(&dense->gradients[wi + j], gradients);
				weights = __PULSE_SIMD_LOAD(dense->weights + wi + j);
				errors = __PULSE_SIMD_LOAD(this->parent->errors + j);
				__PULSE_SIMD_STORE(this->parent->errors + j ,__PULSE_SIMD_MADD(weights, delta, errors));
			}
		}
	}
	else
	{
		for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
		{
			delta = __PULSE_SIMD_SET_ALL(dense->ddeltas[i]);
			for(int j = 0; j < this->n_inputs; j += __PULSE_SIMD_N_PER_CHUNK)
			{
				gradients = __PULSE_SIMD_LOAD(dense->gradients + wi + j);
				inputs = __PULSE_SIMD_LOAD(this->inputs + j);
				gradients = __PULSE_SIMD_MADD(delta, inputs, gradients);
				__PULSE_SIMD_STORE(&dense->gradients[wi + j], gradients);
			}
		}
	}
}

static void _SIMD_FixDense(PULSE_Layer * this, PULSE_HyperArgs args)
{
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	__PULSE_SIMD_DATATYPE baiases, deltas, weights, gradients;
	const __PULSE_SIMD_DATATYPE HYPER = __PULSE_SIMD_SET_ALL(-args.lr/args.batch_size);
	const __PULSE_SIMD_DATATYPE ZERO = __PULSE_SIMD_ZERO();

	for(int i = 0; i < this->n_outputs; i += __PULSE_SIMD_N_PER_CHUNK)
	{
		baiases = __PULSE_SIMD_LOAD(dense->baiases + i);
		deltas = __PULSE_SIMD_LOAD(dense->deltas + i);
		baiases = __PULSE_SIMD_ADD(baiases, __PULSE_SIMD_MUL(deltas, HYPER));
		__PULSE_SIMD_STORE(dense->baiases + i, baiases);
		__PULSE_SIMD_STORE(dense->deltas + i, ZERO);
	}

	for(int i = 0; i < this->n_outputs * this->n_inputs; i += __PULSE_SIMD_N_PER_CHUNK)
	{
		weights = __PULSE_SIMD_LOAD(dense->weights + i);
		gradients = __PULSE_SIMD_LOAD(dense->gradients + i);
		weights = __PULSE_SIMD_ADD(weights, __PULSE_SIMD_MUL(gradients, HYPER));
		__PULSE_SIMD_STORE(dense->weights + i, weights);
		__PULSE_SIMD_STORE(dense->gradients + i, ZERO);
	}
}

#endif

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



PULSE_Layer PULSE_CreateDenseLayer(int n_inputs, int n_outputs, PULSE_ActivationFunction activation_function, PULSE_OptimizationType optimization)
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
#if defined(__PULSE_SIMD_SUPPORTED)
			layer = PULSE_CreateLayer(n_inputs, n_outputs, PULSE_DENSE, activation_function, &_SIMD_FeedDense, &_SIMD_BackDense, &_SIMD_FixDense, &_DestroyDense, optimization);
#else
			printf("ERROR: PULSE Layer SIMD are not supported on this device");
			exit(1);
#endif
			break;
		case PULSE_OPTIMIZATION_GPU_OPENCL:
			printf("ERROR: PULSE Layer GPU are not supported on this device");
			exit(1);

			break;
	}

	layer.layer = dense;
	return layer;
}



