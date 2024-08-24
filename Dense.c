#include "Include/Dense.h"
#include "Include/PULSE_SIMD.h"


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
        PULSE_DataType delta = this->errors[i] * this->outputs[i];
        dense.deltas[i] += delta;
        for(int j = 0; j < this->n_inputs; j++)
        {
            dense.gradients[wi + j] += delta * this->inputs[j];
            if(this->parent != NULL)
                this->parent->errors[j] += dense.weights[wi + j] * delta;
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
    __PULSE_SIMD_DATATYPE delta, errors, gradients, inputs, weights;
    int i = 0, j = 0, wi = 0, J = this->n_inputs - __PULSE_SIMD_N_PER_CHUNK;

    for(i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
    {
        PULSE_DataType cdelta = this->outputs[i] * this->errors[i];
        dense.deltas[i] += cdelta;
        delta = __PULSE_SIMD_SET_ALL(cdelta);
        j = 0;
        while(j < J)
        {
            gradients = __PULSE_SIMD_LOAD(dense.gradients + wi + j);
            inputs = __PULSE_SIMD_LOAD(this->inputs + j);
            gradients = __PULSE_SIMD_MADD(delta, inputs, gradients);
            __PULSE_SIMD_STORE(&dense.gradients[wi + j], gradients);
            if(this->parent != NULL)
            {
                weights = __PULSE_SIMD_LOAD(dense.weights + wi + j);
                errors = __PULSE_SIMD_LOAD(this->parent->errors + j);
                __PULSE_SIMD_STORE(this->parent->errors + j,__PULSE_SIMD_MADD(weights, delta, errors));
            }
            j += __PULSE_SIMD_N_PER_CHUNK;
        }

        for(; j < this->n_inputs; j++)
        {
            dense.gradients[wi + j] += cdelta * this->inputs[j];
            if(this->parent != NULL)
                this->parent->errors[j] += dense.weights[wi + j] * cdelta;
        }

    }
}

#endif


unsigned int PULSE_GetDenseWeightsSize(PULSE_DenseLayerArgs args) {
    return args.n_inputs * args.n_outputs + args.n_outputs;
};

unsigned int PULSE_GetDenseIOSize(PULSE_DenseLayerArgs args) {
    return args.n_inputs + args.n_outputs;
};

unsigned int PULSE_GetDenseFixesSize(PULSE_DenseLayerArgs args) {
    return (args.n_inputs * args.n_outputs) + (2*args.n_outputs);
};

unsigned int PULSE_GetDenseErrorsSize(PULSE_DenseLayerArgs args) {
    return args.n_outputs;
};



static void PULSE_DistributeTrainDenseAllocations(PULSE_Layer * this, PULSE_DataType ** FIXES, PULSE_DataType ** ERRORS) {
    this->layer.DENSE.gradients = *FIXES;
    this->layer.DENSE.deltas = *FIXES + this->n_inputs*this->n_outputs;
    this->errors = *ERRORS;
    *FIXES += PULSE_GetDenseWeightsSize((PULSE_DenseLayerArgs) {
        this->n_inputs, this->n_outputs
    });
    *ERRORS += PULSE_GetDenseErrorsSize((PULSE_DenseLayerArgs) {
        this->n_inputs, this->n_outputs
    });
}

PULSE_Layer PULSE_CreateDenseLayer(PULSE_DenseLayerArgs args, PULSE_DataType *MODEL, PULSE_DataType * IO)
{
    PULSE_DenseLayer dense;

    dense.weights = MODEL;
    dense.baiases = MODEL + (args.n_inputs * args.n_outputs);

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
    layer.mode = &PULSE_DistributeTrainDenseAllocations;

    switch(args.optimization)
    {
    case PULSE_OPTIMIZATION_NONE:
        layer.feed = &_FeedDense;
        layer.back = &_BackDense;
        break;
    case PULSE_OPTIMIZATION_SIMD:
        __PULSE_SIMD_CHECK(layer.feed = &_SIMD_FeedDense);
        __PULSE_SIMD_CHECK(layer.back = &_SIMD_BackDense);
        break;
    case PULSE_OPTIMIZATION_GPU_OPENCL:
        printf("ERROR: PULSE Layer GPU are not supported on this device");
        exit(1);
        break;
    }

    return layer;
}



