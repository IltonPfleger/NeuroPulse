#include "dense.h"
#include "pulse_simd.h"


static void _FeedDense(pulse_layer_t * this)
{
    const int BAIASES_OFFSET = this->n_inputs*this->n_outputs;
    for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs) {
        this->outputs[i] = 0;
        for(int j = 0; j < this->n_inputs; j++)
            this->outputs[i] += this->inputs[j] * this->w[wi + j];
        this->outputs[i] += this->w[BAIASES_OFFSET + i];
    }
    this->activate(this->outputs, this->n_outputs, 0);
}


static void _BackDense(pulse_layer_t * this)
{
    const int BAIASES_OFFSET = this->n_inputs*this->n_outputs;
    this->activate(this->outputs, this->n_outputs, 1);
    for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs) {
        PULSE_DATA delta = this->errors[i] * this->outputs[i];
        this->g[BAIASES_OFFSET + i] += delta;
        for(int j = 0; j < this->n_inputs; j++) {
            this->g[wi + j] += delta * this->inputs[j];
            if(this->prev != NULL)
                this->prev->errors[j] += this->w[wi + j] * delta;
        }
    }
}

#ifdef __PULSE_SIMD_SUPPORTED
static void _SIMD_FeedDense(pulse_layer_t * this)
{
    const int BAIASES_OFFSET = this->n_inputs*this->n_outputs;
    memcpy(this->outputs, this->w + BAIASES_OFFSET, sizeof(PULSE_DATA)*this->n_outputs);
    __PULSE_SIMD_DATATYPE inputs, weights, outputs;
    PULSE_DATA output;
    PULSE_DATA * w_ptr = &(this->w[0]);
    int i, j, J = this->n_inputs - __PULSE_SIMD_N_PER_CHUNK;
    for(i = 0; i < this->n_outputs; i++) {
        outputs = __PULSE_SIMD_ZERO();
        j = 0;
        while(j < J) {
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


static void _SIMD_BackDense(pulse_layer_t * this)
{
    const int BAIASES_OFFSET = this->n_inputs*this->n_outputs;
    this->activate(this->outputs, this->n_outputs, 1);
    __PULSE_SIMD_DATATYPE delta, errors, gradients, inputs, weights;
    int i, j, wi, J = this->n_inputs - __PULSE_SIMD_N_PER_CHUNK;

    if(this->prev != NULL)
        for(i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs) {
            PULSE_DATA cdelta = this->outputs[i] * this->errors[i];
            this->g[BAIASES_OFFSET + i] += cdelta;
            delta = __PULSE_SIMD_SET_ALL(cdelta);
            j = 0;
            while(j < J) {
                gradients = __PULSE_SIMD_LOAD(this->g + wi + j);
                inputs = __PULSE_SIMD_LOAD(this->inputs + j);
                gradients = __PULSE_SIMD_MADD(delta, inputs, gradients);
                __PULSE_SIMD_STORE(&this->g[wi + j], gradients);
                weights = __PULSE_SIMD_LOAD(this->w + wi + j);
                errors = __PULSE_SIMD_LOAD(this->prev->errors + j);
                __PULSE_SIMD_STORE(this->prev->errors + j,__PULSE_SIMD_MADD(weights, delta, errors));
                j += __PULSE_SIMD_N_PER_CHUNK;
            }

            for(; j < this->n_inputs; j++) {
                this->g[wi + j] += cdelta * this->inputs[j];
                this->prev->errors[j] += this->w[wi + j] * cdelta;
            }

        } else
        for(i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs) {
            PULSE_DATA cdelta = this->outputs[i] * this->errors[i];
            this->g[BAIASES_OFFSET + i] += cdelta;
            delta = __PULSE_SIMD_SET_ALL(cdelta);
            j = 0;
            while(j < J) {
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

static void PULSE_DenseRandomize(pulse_layer_t * this)
{
    for(int i = 0; i < this->n_inputs*this->n_outputs; i++)
        this->w[i] = (PULSE_DATA)rand()/(PULSE_DATA)(RAND_MAX)*sqrt(2.0/(PULSE_DATA)(this->n_inputs+this->n_outputs));
}

static void PULSE_DenseFree(pulse_layer_t * this)
{
    PULSE_FREE(this->w);
    PULSE_FREE(this->g);
    PULSE_FREE(this->errors);
    PULSE_FREE(this->inputs);
    PULSE_FREE(this->outputs);
}

pulse_layer_t pulse_create_dense_layer(size_t n_inputs, size_t n_outputs, pulse_activation_fnc_e activation, pulse_optimization_e optimization)
{
    pulse_layer_t layer;
    layer.inputs = NULL;
    layer.outputs = NULL;
    layer.w = NULL;
    layer.g = NULL;
    layer.errors = NULL;
    layer.prev = NULL;
    layer.next = NULL;
    layer.type = PULSE_DENSE;
    layer.optimization = optimization;
    layer.n_inputs = n_inputs;
    layer.n_outputs = n_outputs;
    layer.n_weights = (n_outputs*n_inputs) + n_outputs;
    layer.activate = pulse_get_activation_fnc_ptr(activation);
    layer.free = PULSE_DenseFree;

    layer.inputs = PULSE_ALLOC(sizeof(PULSE_DATA)*n_inputs);
    layer.outputs = PULSE_ALLOC(sizeof(PULSE_DATA)*n_outputs);
    layer.w = PULSE_ALLOC(sizeof(PULSE_DATA)*((n_inputs*n_outputs) + n_outputs));
    layer.errors = PULSE_ALLOC(sizeof(PULSE_DATA)*n_outputs);
    layer.g = PULSE_ALLOC(sizeof(PULSE_DATA)*((n_inputs*n_outputs) + n_outputs));


    switch(optimization) {
        case PULSE_OPTIMIZATION_NONE:
            layer.feed = _FeedDense;
            layer.back = _BackDense;
            break;
        case PULSE_OPTIMIZATION_SIMD:
            __PULSE_SIMD_CHECK(layer.feed = _SIMD_FeedDense);
            __PULSE_SIMD_CHECK(layer.back = _SIMD_BackDense);
            break;
    }

    PULSE_DenseRandomize(&layer);
    return layer;
}
