#include "dense.h"
#include "pulse_simd.h"
#include "opencl/PulseOpenCL.h"


static void _FeedDense(pulse_layer_t * this) {
    const int BAIASES_OFFSET = this->n_inputs*this->n_outputs;
    for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs) {
        this->outputs[i] = 0;
        for(int j = 0; j < this->n_inputs; j++)
            this->outputs[i] += this->inputs[j] * this->w[wi + j];
        this->outputs[i] += this->w[BAIASES_OFFSET + i];
    }
    this->activate(this->outputs, this->n_outputs, 0);
}


static void _BackDense(pulse_layer_t * this) {
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
static void _SIMD_FeedDense(pulse_layer_t * this) {
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


static void _SIMD_BackDense(pulse_layer_t * this) {
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

#ifdef __PULSE_GPU_SUPPORTED
static void _GPU_OPENCL_FeedDense(pulse_layer_t *this) {
    static int SIZE = sizeof(float);
    PULSE_OPENCL_COPY_READ_MEM_TO_GPU(this->inputs, 0, SIZE * this->n_inputs);
    PULSE_OPENCL_COPY_READ_MEM_TO_GPU(this->w, this->n_inputs * SIZE, SIZE * ((this->n_inputs * this->n_outputs) + this->n_outputs));
    PULSE_OPENCL_ENQUEUE_FEEDDENSE(this->n_inputs, this->n_outputs);
    PULSE_OPENCL_GET_WRITE_MEM_TO_HOST(this->outputs, 0, SIZE * this->n_outputs);
    this->activate(this->outputs, this->n_outputs, 0);
}

static void _GPU_OPENCL_BackDense(pulse_layer_t *this) {
    static int SIZE = sizeof(float);
    this->activate(this->outputs, this->n_outputs, 1);
    int READ = 0;
    PULSE_OPENCL_COPY_READ_MEM_TO_GPU(this->inputs, READ, SIZE*this->n_inputs); //Move Inputs To GPU Global Memory
    READ += SIZE*this->n_inputs;
    PULSE_OPENCL_COPY_READ_MEM_TO_GPU(this->outputs, READ, SIZE*this->n_outputs); //Move Outputs To GPU Global Memory
    READ += SIZE*this->n_outputs;
    PULSE_OPENCL_COPY_READ_MEM_TO_GPU(this->errors, READ, SIZE*this->n_outputs); //Move Errors To GPU Global Memory
    READ += SIZE*this->n_outputs;
    PULSE_OPENCL_COPY_READ_MEM_TO_GPU(this->w, READ, SIZE*(this->n_outputs*this->n_inputs + this->n_outputs)); //Move Weights + Baises To GPU Global Memory
    PULSE_OPENCL_COPY_WRITE_MEM_TO_GPU(this->g, 0, SIZE*(this->n_outputs*this->n_inputs + this->n_outputs)); //Move Gradients To GPU Global Memory
    PULSE_OPENCL_ENQUEUE_BACKDENSE(this->n_inputs, this->n_outputs); //Run BackDense Kernel
    PULSE_OPENCL_GET_WRITE_MEM_TO_HOST(this->g, 0, SIZE*(this->n_outputs*this->n_inputs + this->n_outputs)); //Get New Gradients To HOST.

    for(int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs) {
        PULSE_DATA delta = this->errors[i] * this->outputs[i];
        for(int j = 0; j < this->n_inputs; j++) {
            if(this->prev != NULL)
                this->prev->errors[j] += this->w[wi + j] * delta;
        }
    }
}


#endif


static void PULSE_DenseDistributeTrainAllocations(pulse_layer_t * this, PULSE_DATA ** FIXES, PULSE_DATA ** ERRORS) {
    this->g = *FIXES;
    this->errors = *ERRORS;
    *FIXES += this->n_inputs * this->n_outputs + this->n_outputs;
    *ERRORS += this->n_outputs;
}

static void PULSE_DenseDistributeAllocations(pulse_layer_t * this, PULSE_DATA ** WEIGHTS, PULSE_DATA ** IO) {
    this->w = *WEIGHTS;
    this->inputs = *IO;
    this->outputs = *IO  + this->n_inputs;
    *WEIGHTS += this->n_inputs * this->n_outputs + this->n_outputs;
    *IO += this->n_inputs;
}

static void PULSE_DenseRandomize(pulse_layer_t * this) {
    for(int i = 0; i < this->n_inputs*this->n_outputs; i++)
        this->w[i] = (PULSE_DATA)rand()/(PULSE_DATA)(RAND_MAX)*sqrt(2.0/(PULSE_DATA)(this->n_inputs+this->n_outputs));
}

static size_t PULSE_DenseGetWeightsSize(pulse_layer_t * this) {
    return this->n_inputs * this->n_outputs + this->n_outputs;
}

pulse_layer_t pulse_create_dense_layer(size_t n_inputs, size_t n_outputs, pulse_activation_fnc_e activation, pulse_optimization_e optimization) {
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
    layer.activate = pulse_get_activation_fnc_ptr(activation);
    layer.mode = &PULSE_DenseDistributeTrainAllocations;
    layer.start = &PULSE_DenseDistributeAllocations;
    layer.randomize = &PULSE_DenseRandomize;
    layer.get_weights_size = &PULSE_DenseGetWeightsSize;

    switch(optimization) {
        case PULSE_OPTIMIZATION_NONE:
            layer.feed = &_FeedDense;
            layer.back = &_BackDense;
            break;
        case PULSE_OPTIMIZATION_SIMD:
            __PULSE_SIMD_CHECK(layer.feed = &_SIMD_FeedDense);
            __PULSE_SIMD_CHECK(layer.back = &_SIMD_BackDense);
            break;
        case PULSE_OPTIMIZATION_GPU_OPENCL:
            __PULSE_OPENCL_GPU_CHECK(layer.feed = &_GPU_OPENCL_FeedDense);
            __PULSE_OPENCL_GPU_CHECK(layer.back = &_GPU_OPENCL_BackDense);
            __PULSE_OPENCL_GPU_CHECK(PULSE_OPENCL_START());
            break;
    }

    return layer;
}
