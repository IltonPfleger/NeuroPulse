#include "Include/Dense.h"
#include "Include/PULSE_SIMD.h"


static void _FeedDense(PULSE_layer_t * this)
{
    const int B = this->n_inputs*this->n_outputs;
    for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
    {
        this->outputs[i] = 0;
        for(int j = 0; j < this->n_inputs; j++)
            this->outputs[i] += this->inputs[j] * this->w[wi + j];
        this->outputs[i] += this->w[B + i];
    }
    this->activate(this->outputs, this->n_outputs, 0);
}


static void _BackDense(PULSE_layer_t * this)
{
    const int B = this->n_inputs*this->n_outputs;
    this->activate(this->outputs, this->n_outputs, 1);
    for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
    {
        PULSE_data_t delta = this->errors[i] * this->outputs[i];
        this->g[B + i] += delta;
        for(int j = 0; j < this->n_inputs; j++)
        {
            this->g[wi + j] += delta * this->inputs[j];
            if(this->parent != NULL)
                this->parent->errors[j] += this->w[wi + j] * delta;
        }
    }
}

#ifdef __PULSE_SIMD_SUPPORTED
static void _SIMD_FeedDense(PULSE_layer_t * this)
{
    const int B = this->n_inputs*this->n_outputs;
    memcpy(this->outputs, this->w + B, sizeof(PULSE_data_t)*this->n_outputs);
    __PULSE_SIMD_DATATYPE inputs, weights, outputs;
    PULSE_data_t output;
    PULSE_data_t * w_ptr = &(this->w[0]);
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


static void _SIMD_BackDense(PULSE_layer_t * this)
{
    const int B = this->n_inputs*this->n_outputs;
    this->activate(this->outputs, this->n_outputs, 1);
    __PULSE_SIMD_DATATYPE delta, errors, gradients, inputs, weights;
    int i = 0, j = 0, wi = 0, J = this->n_inputs - __PULSE_SIMD_N_PER_CHUNK;

    if(this->parent != NULL)
        for(i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
        {
            PULSE_data_t cdelta = this->outputs[i] * this->errors[i];
            this->g[B + i] += cdelta;
            delta = __PULSE_SIMD_SET_ALL(cdelta);
            j = 0;
            while(j < J)
            {
                gradients = __PULSE_SIMD_LOAD(this->g + wi + j);
                inputs = __PULSE_SIMD_LOAD(this->inputs + j);
                gradients = __PULSE_SIMD_MADD(delta, inputs, gradients);
                __PULSE_SIMD_STORE(&this->g[wi + j], gradients);
                weights = __PULSE_SIMD_LOAD(this->w + wi + j);
                errors = __PULSE_SIMD_LOAD(this->parent->errors + j);
                __PULSE_SIMD_STORE(this->parent->errors + j,__PULSE_SIMD_MADD(weights, delta, errors));
                j += __PULSE_SIMD_N_PER_CHUNK;
            }

            for(; j < this->n_inputs; j++)
            {
                this->g[wi + j] += cdelta * this->inputs[j];
                this->parent->errors[j] += this->w[wi + j] * cdelta;
            }

        }
    else
        for(i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs)
        {
            PULSE_data_t cdelta = this->outputs[i] * this->errors[i];
            this->g[B + i] += cdelta;
            delta = __PULSE_SIMD_SET_ALL(cdelta);
            j = 0;
            while(j < J)
            {
                gradients = __PULSE_SIMD_LOAD(this->g + wi + j);
                inputs = __PULSE_SIMD_LOAD(this->inputs + j);
                gradients = __PULSE_SIMD_MADD(delta, inputs, gradients);
                __PULSE_SIMD_STORE(&this->g[wi + j], gradients);
                j += __PULSE_SIMD_N_PER_CHUNK;
            }

            for(; j < this->n_inputs; j++)
                this->g[wi + j] += cdelta * this->inputs[j];

        }

}

#endif


static void PULSE_DenseDistributeTrainAllocations(PULSE_layer_t * this, PULSE_data_t ** FIXES, PULSE_data_t ** ERRORS)
{
    this->g = *FIXES;
    this->errors = *ERRORS;
    *FIXES += this->n_inputs * this->n_outputs + this->n_outputs;
    *ERRORS += this->n_outputs;
}

static void PULSE_DenseDistributeAllocations(PULSE_layer_t * this, PULSE_data_t ** WEIGHTS, PULSE_data_t ** IO)
{
    this->w = *WEIGHTS;
    this->inputs = *IO;
    this->outputs = *IO  + this->n_inputs;
    *WEIGHTS += this->n_inputs * this->n_outputs + this->n_outputs;
    *IO += this->n_outputs + this->n_inputs;
}

static void PULSE_DenseRandomize(PULSE_layer_t * this)
{
    for(int i = 0; i < this->n_inputs*this->n_outputs; i++)
        this->w[i] = (PULSE_data_t)rand()/(PULSE_data_t)(RAND_MAX)*sqrt(2.0/(PULSE_data_t)(this->n_inputs+this->n_outputs));
}



PULSE_layer_t PULSE_CreateDenseLayer(PULSE_DenseLayerArgs args)
{
    PULSE_layer_t layer;
    layer.inputs = NULL;
    layer.outputs = NULL;
    layer.w = NULL;
    layer.g = NULL;
    layer.errors = NULL;
    layer.parent = NULL;
    layer.child = NULL;
    layer.type = PULSE_DENSE;
    layer.optimization = args.optimization;
    layer.n_inputs = args.n_inputs;
    layer.n_outputs = args.n_outputs;
    layer.activate = PULSE_GetActivationFunctionPtr(args.activation_function);
    layer.mode = &PULSE_DenseDistributeTrainAllocations;
    layer.start = &PULSE_DenseDistributeAllocations;
    layer.randomize = &PULSE_DenseRandomize;

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



