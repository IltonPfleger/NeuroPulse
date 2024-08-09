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

//INTEL AVX
#if defined(__PULSE_SIMD_AVX)
static void _SIMD_FeedDense(PULSE_Layer * this)
{
	static const int CHUNK_SIZE = 256/sizeof(__m256);
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	__m256 inputs, weights, outputs;
	for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
	{
		outputs = _mm256_setzero_ps();
		for(int j = 0; j < this->n_inputs; j += CHUNK_SIZE)
		{
			weights = _mm256_loadu_ps(dense->weights + wi + j);
			inputs = _mm256_loadu_ps(this->inputs + j);
			if(j + CHUNK_SIZE > this->n_inputs)
			{
				weights = __PULSE_SIMD_X86_ZERO_R_256(weights, CHUNK_SIZE - this->n_inputs - j);
				inputs = __PULSE_SIMD_X86_ZERO_R_256(inputs, CHUNK_SIZE - this->n_inputs - j);
			}
			outputs = _mm256_fmadd_ps(weights, inputs, outputs);
		}
		this->outputs[i] = _mm_cvtss_f32(__PULSE_SIMD_X86_REDUCE_ADD_256(outputs)) + dense->baiases[i];
	}
	this->activate(this, 0);
}


static void _SIMD_BackDense(PULSE_Layer * this)
{
	static const int CHUNK_SIZE = 256/sizeof(__m256);
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	this->activate(this, 1);
	__m256 deltas, ddeltas, delta, errors, gradients, inputs, weights, outputs;

	for(int i = 0; i < this->n_outputs; i += CHUNK_SIZE)
	{
		errors = _mm256_loadu_ps(this->errors + i);
		outputs = _mm256_loadu_ps(this->outputs + i);
		ddeltas = _mm256_loadu_ps(dense->deltas + i);
		deltas = _mm256_loadu_ps(dense->ddeltas + i);
		ddeltas = _mm256_mul_ps(errors, outputs);
		deltas = _mm256_add_ps(ddeltas, deltas);
		_mm256_storeu_ps(dense->ddeltas + i, ddeltas);
		_mm256_storeu_ps(dense->deltas + i, deltas);
	};

	for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
	{
		delta = _mm256_set1_ps(dense->ddeltas[i]);
		for(int j = 0; j < this->n_inputs; j += CHUNK_SIZE)
		{
			gradients = _mm256_loadu_ps(dense->gradients + wi + j);
			inputs = _mm256_loadu_ps(this->inputs + j);
			gradients = _mm256_fmadd_ps(delta, inputs, gradients);
			_mm256_storeu_ps(&dense->gradients[wi + j], gradients);

			if(this->parent != NULL)
			{
				weights = _mm256_loadu_ps(dense->weights + wi + j);
				errors = _mm256_loadu_ps(this->parent->errors + j);
				_mm256_storeu_ps(this->parent->errors + j ,_mm256_fmadd_ps(weights, delta, errors));
			}
		}
	}
}

static void _SIMD_FixDense(PULSE_Layer * this, PULSE_HyperArgs args)
{
	static const int CHUNK_SIZE = 256/sizeof(__m256);
	PULSE_DenseLayer * dense = (PULSE_DenseLayer*)this->layer;
	__m256 baiases, deltas, weights, gradients;
	const __m256 HYPER = _mm256_set1_ps(-args.lr/args.batch_size);
	const __m256 ZERO = _mm256_setzero_ps();

	for(int i = 0; i < this->n_outputs; i += CHUNK_SIZE)
	{
		baiases = _mm256_loadu_ps(dense->baiases + i);
		deltas = _mm256_loadu_ps(dense->deltas + i);
		baiases = _mm256_add_ps(baiases, _mm256_mul_ps(deltas, HYPER));
		_mm256_storeu_ps(dense->baiases + i, baiases);
		_mm256_storeu_ps(dense->deltas + i, ZERO);
	}

	for(int i = 0; i < this->n_outputs * this->n_inputs; i += CHUNK_SIZE)
	{
		weights = _mm256_loadu_ps(dense->weights + i);
		gradients = _mm256_loadu_ps(dense->gradients + i);
		weights = _mm256_add_ps(weights, _mm256_mul_ps(gradients, HYPER));
		_mm256_storeu_ps(dense->weights + i, weights);
		_mm256_storeu_ps(dense->gradients + i, ZERO);
	}
}

#endif//INTEL AVX

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
#if defined(__PULSE_SIMD_SUPPORTED)
			layer = PULSE_CreateLayer(n_inputs, n_outputs, PULSE_DENSE, activation_function, &_SIMD_FeedDense, &_SIMD_BackDense, &_SIMD_FixDense, &_DestroyDense, optimization);
#else
			printf("ERROR: PULSE SIMD are not supported on this device");
			exit(1);
#endif
			break;
	}

	layer.layer = dense;
	return layer;
}



