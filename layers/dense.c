#include <activations/activations.h>
#include <layers/dense.h>
#include <memory/memory.h>

static void feed_dense(pulse_layer_t* this) {
    pulse_dense_layer_t dense = *(pulse_dense_layer_t*)this->internal;
    for (int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs) {
        this->outputs[i] = 0;
        for (int j = 0; j < this->n_inputs; j++) {
            this->outputs[i] += this->inputs[j] * dense.w[wi + j];
        }
        this->outputs[i] += dense.b[i];
    }
    dense.activate(this->outputs, this->n_outputs, 0);
}

static void back_dense(pulse_layer_t* this) {
    pulse_dense_layer_t dense = *(pulse_dense_layer_t*)this->internal;
    dense.activate(this->outputs, this->n_outputs, 1);
    for (int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs) {
        dense.d[i] += this->errors[i] * this->outputs[i];
        for (int j = 0; j < this->n_inputs; j++) {
            dense.g[wi + j] += dense.d[i] * this->inputs[j];
            if (this->prev != NULL) {
                this->prev->errors[j] += dense.w[wi + j] * dense.d[i];
            }
        }
    }
}

// #ifdef PULSE_SIMD_SUPPORTED
// static void _simd_feed_dense(pulse_layer_t* this) {
//     const int BAIASES_OFFSET = this->n_inputs * this->n_outputs;
//     memcpy(this->outputs, this->w + BAIASES_OFFSET, sizeof(pulse_datatype) * this->n_outputs);
//     PULSE_SIMD_DATATYPE inputs, weights, outputs;
//     pulse_datatype output;
//     pulse_datatype* w_ptr = &(this->w[0]);
//     int i, j, J = this->n_inputs - PULSE_SIMD_N_PER_CHUNK;
//     for (i = 0; i < this->n_outputs; i++) {
//         outputs = PULSE_SIMD_ZERO();
//         j       = 0;
//         while (j < J) {
//             weights = PULSE_SIMD_ALLIGNED_LOAD(w_ptr);
//             inputs  = PULSE_SIMD_ALLIGNED_LOAD(this->inputs + j);
//             outputs = PULSE_SIMD_MADD(weights, inputs, outputs);
//             j += PULSE_SIMD_N_PER_CHUNK;
//             w_ptr += PULSE_SIMD_N_PER_CHUNK;
//         }
//         output = PULSE_SIMD_TO_FLOAT(PULSE_SIMD_REDUCE_ADD(outputs));
//         for (; j < this->n_inputs; j++, w_ptr++) output += *w_ptr * this->inputs[j];
//         this->outputs[i] += output;
//     }
//     this->activate(this->outputs, this->n_outputs, 0);
// }
//
// static void _simd_back_dense(pulse_layer_t* this) {
//     const int BAIASES_OFFSET = this->n_inputs * this->n_outputs;
//     this->activate(this->outputs, this->n_outputs, 1);
//     PULSE_SIMD_DATATYPE delta, errors, gradients, inputs, weights;
//     int i, j, wi, J = this->n_inputs - PULSE_SIMD_N_PER_CHUNK;
//
//     if (this->prev != NULL)
//         for (i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs) {
//             pulse_datatype cdelta = this->outputs[i] * this->errors[i];
//             this->g[BAIASES_OFFSET + i] += cdelta;
//             delta = PULSE_SIMD_SET_ALL(cdelta);
//             j     = 0;
//             while (j < J) {
//                 gradients = PULSE_SIMD_LOAD(this->g + wi + j);
//                 inputs    = PULSE_SIMD_LOAD(this->inputs + j);
//                 gradients = PULSE_SIMD_MADD(delta, inputs, gradients);
//                 PULSE_SIMD_STORE(&this->g[wi + j], gradients);
//                 weights = PULSE_SIMD_LOAD(this->w + wi + j);
//                 errors  = PULSE_SIMD_LOAD(this->prev->errors + j);
//                 PULSE_SIMD_STORE(this->prev->errors + j, PULSE_SIMD_MADD(weights, delta, errors));
//                 j += PULSE_SIMD_N_PER_CHUNK;
//             }
//
//             for (; j < this->n_inputs; j++) {
//                 this->g[wi + j] += cdelta * this->inputs[j];
//                 this->prev->errors[j] += this->w[wi + j] * cdelta;
//             }
//         }
//     else {
//         for (i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs) {
//             pulse_datatype cdelta = this->outputs[i] * this->errors[i];
//             this->g[BAIASES_OFFSET + i] += cdelta;
//             delta = PULSE_SIMD_SET_ALL(cdelta);
//             j     = 0;
//             while (j < J) {
//                 gradients = PULSE_SIMD_LOAD(this->g + wi + j);
//                 inputs    = PULSE_SIMD_LOAD(this->inputs + j);
//                 gradients = PULSE_SIMD_MADD(delta, inputs, gradients);
//                 PULSE_SIMD_STORE(&this->g[wi + j], gradients);
//                 j += PULSE_SIMD_N_PER_CHUNK;
//             }
//
//             for (; j < this->n_inputs; j++) this->g[wi + j] += cdelta * this->inputs[j];
//         }
//     }
// }
// #endif

static void fix_dense(pulse_layer_t* this, double HYPER) {
    pulse_dense_layer_t* dense = (pulse_dense_layer_t*)this->internal;
    for (int i = 0, wi = 0; i < this->n_outputs; i++, wi += this->n_inputs) {
        dense->b[i] += dense->d[i] * HYPER;
        dense->d[i] = 0;
        for (int j = 0; j < this->n_inputs; j++) {
            dense->w[wi + j] += dense->g[wi + j] * HYPER;
            dense->g[wi + j] = 0;
        }
    }
}

static void free_dense(pulse_layer_t* this) {
    //     PULSE_FREE(this->w);
    //     PULSE_FREE(this->g);
    //     PULSE_FREE(this->errors);
    //     PULSE_FREE(this->inputs);
    //     PULSE_FREE(this->outputs);
}

pulse_layer_t pulse_create_dense_layer(size_t n_inputs, size_t n_outputs, pulse_activation_function activation) {
    pulse_layer_t layer;
    pulse_dense_layer_t* dense = pulse_memory_alloc(sizeof(pulse_dense_layer_t));

    layer.n_inputs  = n_inputs;
    layer.n_outputs = n_outputs;

    // layer.inputs       = NULL;
    // layer.outputs      = NULL;
    // layer.w            = NULL;
    // layer.g            = NULL;
    // layer.errors       = NULL;
    // layer.prev         = NULL;
    // layer.next         = NULL;
    // layer.type         = PULSE_DENSE;
    // layer.optimization = optimization;
    // layer.n_weights    = (n_outputs * n_inputs) + n_outputs;
    // layer.activate     = pulse_get_activation_fnc_ptr(activation);

    layer.feed = feed_dense;
    layer.back = back_dense;
    layer.free = free_dense;
    layer.fix  = fix_dense;

    layer.inputs    = pulse_memory_alloc(sizeof(pulse_datatype) * n_inputs);
    layer.outputs   = pulse_memory_alloc(sizeof(pulse_datatype) * n_outputs);
    layer.errors    = pulse_memory_alloc(sizeof(pulse_datatype) * n_outputs);
    dense->w        = pulse_memory_alloc(sizeof(pulse_datatype) * ((n_inputs * n_outputs) + n_outputs));
    dense->g        = pulse_memory_alloc(sizeof(pulse_datatype) * ((n_inputs * n_outputs) + n_outputs));
    dense->d        = pulse_memory_alloc(sizeof(pulse_datatype) * n_outputs);
    dense->b        = pulse_memory_alloc(sizeof(pulse_datatype) * n_outputs);
    dense->activate = activation;

    layer.internal = dense;

    for (int i = 0; i < n_inputs * n_outputs; i++)
        dense->w[i] = (pulse_datatype)rand() / (pulse_datatype)(RAND_MAX)*sqrt(2.0 / (pulse_datatype)(n_inputs + n_outputs));
    return layer;
}
