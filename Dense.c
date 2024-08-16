#include "Include/Dense.h"
#include "Include/PULSE_SIMD.h"


//STD
static void _FeedDense(PULSE_Layer * this)
{
	PULSE_DenseLayer dense = this->layer.DENSE;
	for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
	{
		this->outputs[i] = 0;
		for(int j = 0; j < this->n_inputs; j++)
			this->outputs[i] += this->inputs[j] * dense.weights[wi + j];
		this->outputs[i] += dense.baiases[i];
	}
	this->activate(this->outputs, this->n_outputs, 0);
}


static void _BackDense(PULSE_Layer * this)
{
	PULSE_DenseLayer dense = this->layer.DENSE;
	this->activate(this->outputs, this->n_outputs, 1);
	for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
	{
		double delta = this->errors[i] * this->outputs[i];
		dense.deltas[i] += delta;
		for(int j = 0; j < this->n_inputs; j++)
		{
			dense.gradients[wi + j] += delta * this->inputs[j];
			if(this->parent != NULL)
				this->parent->errors[j] += dense.weights[wi + j] * delta;
		}
	}
}

static void _FixDense(PULSE_Layer * this, PULSE_HyperArgs args)
{
	PULSE_DenseLayer dense = this->layer.DENSE;
	const double HYPER = args.lr/args.batch_size;
	for (int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
	{
		dense.baiases[i] -= HYPER * dense.deltas[i];
		dense.deltas[i] = 0;
		for (int j = 0; j < this->n_inputs; j++)
		{
			dense.weights[wi + j] -= HYPER * dense.gradients[wi + j];
			dense.gradients[wi + j] = 0;
		}
	}
}

#ifdef __PULSE_SIMD_SUPPORTED
static void _SIMD_FeedDense(PULSE_Layer * this)
{
	PULSE_DenseLayer dense = this->layer.DENSE;
	memcpy(this->outputs, dense.baiases, sizeof(PULSE_DataType)*this->n_outputs);
	__PULSE_SIMD_DATATYPE inputs, weights, outputs;
	PULSE_DataType output;
	PULSE_DataType * w_ptr = &(dense.weights[0]);
	int i, j, J = this->n_inputs - __PULSE_SIMD_N_PER_CHUNK;
	for(i = 0; i < this->n_outputs; i++)
	{
		outputs = __PULSE_SIMD_ZERO();
		j = 0;
		while(j < J)
		{
			weights = __PULSE_SIMD_ALLIGNED_LOAD(w_ptr);
			inputs = __PULSE_SIMD_ALLIGNED_LOAD(this->inputs + j);
			outputs = __PULSE_SIMD_MADD(weights, inputs, outputs);
			j += __PULSE_SIMD_N_PER_CHUNK;
			w_ptr += __PULSE_SIMD_N_PER_CHUNK;
		}
		output = __PULSE_SIMD_TO_FLOAT(__PULSE_SIMD_REDUCE_ADD(outputs));
		for(; j < this->n_inputs; j++, w_ptr++)
			output += *w_ptr * this->inputs[j];
		this->outputs[i] += output;
	}
	this->activate(this->outputs, this->n_outputs, 0);
}


static void _SIMD_BackDense(PULSE_Layer * this)
{
	PULSE_DenseLayer dense = this->layer.DENSE;
	this->activate(this->outputs, this->n_outputs, 1);
	__PULSE_SIMD_DATATYPE deltas, ddeltas, delta, errors, gradients, inputs, weights, outputs;
	int i = 0, j = 0, wi = 0;

	for(i = 0; i < this->n_outputs; i += __PULSE_SIMD_N_PER_CHUNK)
	{
		errors = __PULSE_SIMD_LOAD(this->errors + i);
		outputs = __PULSE_SIMD_LOAD(this->outputs + i);
		ddeltas = __PULSE_SIMD_LOAD(dense.deltas + i);
		deltas = __PULSE_SIMD_LOAD(dense.ddeltas + i);
		ddeltas = __PULSE_SIMD_MUL(errors, outputs);
		deltas = __PULSE_SIMD_ADD(ddeltas, deltas);
		__PULSE_SIMD_STORE(dense.ddeltas + i, ddeltas);
		__PULSE_SIMD_STORE(dense.deltas + i, deltas);
	};


	if(this->parent != NULL)
	{
		for(i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
		{
			delta = __PULSE_SIMD_SET_ALL(dense.ddeltas[i]);
			for(j = 0; j < this->n_inputs; j += __PULSE_SIMD_N_PER_CHUNK)
			{
				gradients = __PULSE_SIMD_LOAD(dense.gradients + wi + j);
				inputs = __PULSE_SIMD_LOAD(this->inputs + j);
				gradients = __PULSE_SIMD_MADD(delta, inputs, gradients);
				__PULSE_SIMD_STORE(&dense.gradients[wi + j], gradients);
				weights = __PULSE_SIMD_LOAD(dense.weights + wi + j);
				errors = __PULSE_SIMD_LOAD(this->parent->errors + j);
				__PULSE_SIMD_STORE(this->parent->errors + j ,__PULSE_SIMD_MADD(weights, delta, errors));
			}
		}
	}
	else
	{
		for(i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
		{
			delta = __PULSE_SIMD_SET_ALL(dense.ddeltas[i]);
			for(j = 0; j < this->n_inputs; j += __PULSE_SIMD_N_PER_CHUNK)
			{
				gradients = __PULSE_SIMD_LOAD(dense.gradients + wi + j);
				inputs = __PULSE_SIMD_LOAD(this->inputs + j);
				gradients = __PULSE_SIMD_MADD(delta, inputs, gradients);
				__PULSE_SIMD_STORE(&dense.gradients[wi + j], gradients);
			}
		}
	}
}

static void _SIMD_FixDense(PULSE_Layer * this, PULSE_HyperArgs args)
{
	PULSE_DenseLayer dense = this->layer.DENSE;
	__PULSE_SIMD_DATATYPE baiases, deltas, weights, gradients;
	const __PULSE_SIMD_DATATYPE ZERO = __PULSE_SIMD_ZERO();
	const __PULSE_SIMD_DATATYPE HYPER = __PULSE_SIMD_SET_ALL(-args.lr/args.batch_size);
	const int I_DELTAS_SIZE = this->n_outputs - __PULSE_SIMD_N_PER_CHUNK;
	const int I_WEIGHTS_SIZE = this->n_inputs*this->n_outputs - __PULSE_SIMD_N_PER_CHUNK;
	int i = 0;
	while(i < I_DELTAS_SIZE)
	{
		baiases = __PULSE_SIMD_ALLIGNED_LOAD(dense.baiases + i);
		deltas = __PULSE_SIMD_ALLIGNED_LOAD(dense.deltas + i);
		baiases = __PULSE_SIMD_ADD(baiases, __PULSE_SIMD_MUL(deltas, HYPER));
		__PULSE_SIMD_STREAM_STORE(dense.baiases + i, baiases);
		__PULSE_SIMD_STREAM_STORE(dense.deltas + i, ZERO);
		i += __PULSE_SIMD_N_PER_CHUNK;
	};
	for(; i < this->n_outputs; i++)
	{
		dense.baiases[i] += dense.deltas[i];
		dense.deltas[i] = 0;
	}

	i = 0;
	while(i < I_WEIGHTS_SIZE)
	{
		weights = __PULSE_SIMD_ALLIGNED_LOAD(dense.weights + i);
		gradients = __PULSE_SIMD_ALLIGNED_LOAD(dense.gradients + i);
		weights = __PULSE_SIMD_ADD(weights, __PULSE_SIMD_MUL(gradients, HYPER));
		__PULSE_SIMD_STREAM_STORE(dense.weights + i, weights);
		__PULSE_SIMD_STREAM_STORE(dense.gradients + i, ZERO);
		i += __PULSE_SIMD_N_PER_CHUNK;
	};
	for(; i < this->n_outputs; i++)
	{
		dense.weights[i] += dense.gradients[i];
		dense.gradients[i] = 0;
	}
}
#endif


unsigned int PULSE_GetDenseWeightsSize(PULSE_DenseLayerArgs args) { return args.n_inputs * args.n_outputs + args.n_outputs; };
unsigned int PULSE_GetDenseIOSize(PULSE_DenseLayerArgs args) { return args.n_inputs + args.n_outputs; };

PULSE_Layer PULSE_CreateDenseLayer(PULSE_DenseLayerArgs args, PULSE_DataType *MODEL, PULSE_DataType * IO)
{
	PULSE_DenseLayer dense;

	dense.weights = MODEL;
	dense.baiases = MODEL + (args.n_inputs * args.n_outputs);
	dense.gradients = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*args.n_inputs*args.n_outputs);
	dense.deltas = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*args.n_outputs);
	dense.ddeltas = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*args.n_outputs);

	for(int i = 0; i < args.n_inputs*args.n_outputs; i++)
		dense.weights[i] = (PULSE_DataType)rand()/(PULSE_DataType)(RAND_MAX)*sqrt(2.0/(PULSE_DataType)(args.n_inputs+args.n_outputs));

	PULSE_Layer layer;
	layer.inputs = IO;
	layer.outputs = IO + args.n_inputs;;
	layer.errors = NULL;
	layer.type = PULSE_DENSE;
	layer.optimization = args.optimization;
	layer.parent = NULL;
	layer.child = NULL;
	layer.layer.DENSE = dense;
	layer.n_inputs = args.n_inputs;
	layer.n_outputs = args.n_outputs;
	layer.activate = PULSE_GetActivationFunctionPtr(args.activation_function);

	layer.errors = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*args.n_outputs); //REMOVERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR


	switch(args.optimization)
	{
		case PULSE_OPTIMIZATION_NONE:
			layer.feed = &_FeedDense;
			layer.back = &_BackDense;
			layer.fix = &_FixDense;
			break;
		case PULSE_OPTIMIZATION_SIMD:
			__PULSE_SIMD_CHECK(layer.feed = &_SIMD_FeedDense);
			__PULSE_SIMD_CHECK(layer.back = &_SIMD_BackDense);
			__PULSE_SIMD_CHECK(layer.fix = &_SIMD_FixDense);
			break;
		case PULSE_OPTIMIZATION_GPU_OPENCL:
			printf("ERROR: PULSE Layer GPU are not supported on this device");
			exit(1);
			break;
	}

	return layer;
}



