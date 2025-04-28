#ifndef PULSE_DENSE_H
#define PULSE_DENSE_H

#include <activations/activation.h>
#include <assert.h>
#include <debug/debug.h>
#include <layers/layer.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    void *w;
    void *b;
    void *g;
    void *d;
    pulse_activation_function activate;
} pulse_dense_layer_t;

#define FEED(dtype)                                                         \
    static void feed_##dtype(pulse_layer_t *this, void *ptr)                \
    {                                                                       \
        pulse_dense_layer_t dense = *(pulse_dense_layer_t *)this->internal; \
        this->inputs              = ptr;                                    \
        dtype *weights            = (dtype *)dense.w;                       \
        dtype *bias               = (dtype *)dense.b;                       \
        dtype *inputs             = (dtype *)this->inputs;                  \
        dtype *outputs            = (dtype *)this->outputs;                 \
        size_t isize              = this->isize;                            \
        size_t osize              = this->osize;                            \
        for (size_t i = 0, wi = 0; i < osize; i++, wi += isize) {           \
            outputs[i] = bias[i];                                           \
            for (size_t j = 0; j < isize; j++) {                            \
                outputs[i] += inputs[j] * weights[wi + j];                  \
            }                                                               \
            dense.activate(outputs + i, 0);                                 \
        }                                                                   \
        if (this->next) this->next->feed(this->next, outputs);              \
    }

#define BACK(dtype)                                                         \
    static void back_##dtype(pulse_layer_t *this)                           \
    {                                                                       \
        pulse_dense_layer_t dense = *(pulse_dense_layer_t *)this->internal; \
        dtype *weights            = (dtype *)dense.w;                       \
        dtype *gradients          = (dtype *)dense.g;                       \
        dtype *deltas             = (dtype *)dense.d;                       \
        dtype *outputs            = (dtype *)this->outputs;                 \
        dtype *inputs             = (dtype *)this->inputs;                  \
        dtype *errors             = (dtype *)this->errors;                  \
        size_t isize              = this->isize;                            \
        size_t osize              = this->osize;                            \
        if (this->prev) {                                                   \
            dtype *perrors = (dtype *)this->prev->errors;                   \
            for (size_t i = 0, wi = 0; i < osize; i++, wi += isize) {       \
                dense.activate(outputs + i, 1);                             \
                deltas[i] += errors[i] * outputs[i];                        \
                errors[i] = 0;                                              \
                for (size_t j = 0; j < isize; j++) {                        \
                    gradients[wi + j] += deltas[i] * inputs[j];             \
                    perrors[j] += weights[wi + j] * deltas[i];              \
                }                                                           \
            }                                                               \
            this->prev->back(this->prev);                                   \
        } else {                                                            \
            for (size_t i = 0, wi = 0; i < osize; i++, wi += isize) {       \
                deltas[i] += errors[i] * outputs[i];                        \
                errors[i] = 0;                                              \
                for (size_t j = 0; j < isize; j++) {                        \
                    gradients[wi + j] += deltas[i] * inputs[j];             \
                }                                                           \
            }                                                               \
        }                                                                   \
    }

#define FIX(dtype)                                                          \
    static void fix_##dtype(pulse_layer_t *this, pulse_train_args_t args)   \
    {                                                                       \
        pulse_dense_layer_t dense = *(pulse_dense_layer_t *)this->internal; \
                                                                            \
        dtype *weights   = (dtype *)dense.w;                                \
        dtype *bias      = (dtype *)dense.b;                                \
        dtype *gradients = (dtype *)dense.g;                                \
        dtype *deltas    = (dtype *)dense.d;                                \
        size_t isize     = this->isize;                                     \
        size_t osize     = this->osize;                                     \
        double HYPER     = -args.lr / args.batch_size;                      \
        for (size_t i = 0, wi = 0; i < osize; i++, wi += isize) {           \
            bias[i] += deltas[i] * HYPER;                                   \
            deltas[i] = 0;                                                  \
            for (size_t j = 0; j < isize; j++) {                            \
                weights[wi + j] += gradients[wi + j] * HYPER;               \
                gradients[wi + j] = 0;                                      \
            }                                                               \
        }                                                                   \
        if (this->next) {                                                   \
            this->next->fix(this->next, args);                              \
        }                                                                   \
    }

FEED(int)
FEED(float)
FEED(double)

BACK(int)
BACK(float)
BACK(double)

FIX(int)
FIX(float)
FIX(double)

static void (*PULSE_DENSE_FEED[])(pulse_layer_t *, void *)            = {[PULSE_INT] = feed_int, [PULSE_FLOAT] = feed_float, [PULSE_DOUBLE] = feed_double};
static void (*PULSE_DENSE_BACK[])(pulse_layer_t *)                    = {[PULSE_INT] = back_int, [PULSE_FLOAT] = back_float, [PULSE_DOUBLE] = back_double};
static void (*PULSE_DENSE_FIX[])(pulse_layer_t *, pulse_train_args_t) = {[PULSE_INT] = fix_int, [PULSE_FLOAT] = fix_float, [PULSE_DOUBLE] = fix_double};

static void PULSE_DENSE_FREE(pulse_layer_t *this)
{
    pulse_dense_layer_t dense = *(pulse_dense_layer_t *)this->internal;
    free(this->outputs);
    free(this->errors);
    free(dense.w);
    free(dense.b);
    free(dense.g);
    free(dense.d);
    free(this->internal);
};

pulse_layer_t pulse_dense_layer(size_t isize, size_t osize, pulse_dtype_t dtype, pulse_activation_function activate)
{
    srand(time(NULL));
    const size_t DTYPE_SIZE = pulse_dtype_sizes[dtype];

    pulse_dense_layer_t *dense_ptr = (pulse_dense_layer_t *)malloc(sizeof(pulse_dense_layer_t));
    void *outputs                  = malloc(DTYPE_SIZE * osize);
    void *errors                   = malloc(DTYPE_SIZE * osize);
    void *weights                  = malloc(DTYPE_SIZE * osize * isize);
    void *gradients                = malloc(DTYPE_SIZE * osize * isize);
    void *biases                   = malloc(DTYPE_SIZE * osize);
    void *deltas                   = malloc(DTYPE_SIZE * osize);

    PULSE_DEBUG_ERROR(isize <= 0, "PulseDenseLayer::Create >> Input length can't be negative or zero.");
    PULSE_DEBUG_ERROR(osize <= 0, "PulseDenseLayer::Create >> Output length can't be negative or zero.");
    PULSE_DEBUG_ERROR(activate == NULL, "PulseDenseLayer::Create >> Activation function can't be NULL.");
    PULSE_DEBUG_ERROR(dense_ptr == NULL, "PulseDenseLayer::Create >> Heap memory allocation failed.");
    PULSE_DEBUG_ERROR(outputs == NULL, "PulseDenseLayer::Create >> Heap memory allocation failed.");
    PULSE_DEBUG_ERROR(errors == NULL, "PulseDenseLayer::Create >> Heap memory allocation failed.");
    PULSE_DEBUG_ERROR(weights == NULL, "PulseDenseLayer::Create >> Heap memory allocation failed.");
    PULSE_DEBUG_ERROR(biases == NULL, "PulseDenseLayer::Create >> Heap memory allocation failed.");
    PULSE_DEBUG_ERROR(deltas == NULL, "PulseDenseLayer::Create >> Heap memory allocation failed.");

    const pulse_dense_layer_t dense = {
        .w        = weights,
        .g        = gradients,
        .d        = deltas,
        .b        = biases,
        .activate = activate,
    };

    const pulse_layer_t layer = {
        .outputs  = outputs,
        .errors   = errors,
        .prev     = NULL,
        .next     = NULL,
        .feed     = PULSE_DENSE_FEED[dtype],
        .back     = PULSE_DENSE_BACK[dtype],
        .fix      = PULSE_DENSE_FIX[dtype],
        .free     = PULSE_DENSE_FREE,
        .osize    = osize,
        .isize    = isize,
        .internal = dense_ptr,
    };

    memcpy(dense_ptr, &dense, sizeof(pulse_dense_layer_t));

    double *$weights = (double *)dense.w;
    for (size_t i = 0; i < isize * osize; i++) $weights[i] = (double)rand() / (double)(RAND_MAX)*sqrt(2.0 / (double)(isize + osize));
    return layer;
}

#endif
