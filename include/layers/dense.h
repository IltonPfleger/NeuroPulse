#ifndef PULSE_LAYER_DENSE_H
#define PULSE_LAYER_DENSE_H

#include <activations/activation.h>
#include <debug/debug.h>
#include <layers/layer.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct pulse_layer_dense_internal_s {
    void *w;
    void *b;
    void *g;
    void *d;
    pulse_activation_t activate;
    double lr;
};

#define $DENSE_FORWARD(dtype)                                                                                \
    static void forward_##dtype(pulse_layer_t *layer, const void *const ptr)                                 \
    {                                                                                                        \
        struct pulse_layer_dense_internal_s dense = *(struct pulse_layer_dense_internal_s *)layer->internal; \
        layer->inputs                             = (void *const)ptr;                                        \
        dtype *weights                            = (dtype *)dense.w;                                        \
        dtype *bias                               = (dtype *)dense.b;                                        \
        dtype *inputs                             = (dtype *)layer->inputs;                                  \
        dtype *outputs                            = (dtype *)layer->outputs;                                 \
        size_t isize                              = layer->isize;                                            \
        size_t osize                              = layer->osize;                                            \
        for (size_t i = 0, wi = 0; i < osize; i++, wi += isize) {                                            \
            outputs[i] = bias[i];                                                                            \
            for (size_t j = 0; j < isize; j++) {                                                             \
                outputs[i] += inputs[j] * weights[wi + j];                                                   \
            }                                                                                                \
            dense.activate(outputs + i, 0);                                                                  \
        }                                                                                                    \
        if (layer->next) layer->next->forward(layer->next, outputs);                                         \
    }

#define $DENSE_THINK(dtype)                                                                                  \
    static void think_##dtype(pulse_layer_t *layer)                                                          \
    {                                                                                                        \
        struct pulse_layer_dense_internal_s dense = *(struct pulse_layer_dense_internal_s *)layer->internal; \
        dtype *weights                            = (dtype *)dense.w;                                        \
        dtype *gradients                          = (dtype *)dense.g;                                        \
        dtype *deltas                             = (dtype *)dense.d;                                        \
        dtype *outputs                            = (dtype *)layer->outputs;                                 \
        dtype *inputs                             = (dtype *)layer->inputs;                                  \
        dtype *errors                             = (dtype *)layer->errors;                                  \
        size_t isize                              = layer->isize;                                            \
        size_t osize                              = layer->osize;                                            \
        if (layer->prev) {                                                                                   \
            dtype *perrors = (dtype *)layer->prev->errors;                                                   \
            for (size_t i = 0, wi = 0; i < osize; i++, wi += isize) {                                        \
                dense.activate(outputs + i, 1);                                                              \
                deltas[i] += errors[i] * outputs[i];                                                         \
                errors[i] = 0;                                                                               \
                for (size_t j = 0; j < isize; j++) {                                                         \
                    gradients[wi + j] += deltas[i] * inputs[j];                                              \
                    perrors[j] += weights[wi + j] * deltas[i];                                               \
                }                                                                                            \
            }                                                                                                \
            layer->prev->think(layer->prev);                                                                 \
        } else {                                                                                             \
            for (size_t i = 0, wi = 0; i < osize; i++, wi += isize) {                                        \
                deltas[i] += errors[i] * outputs[i];                                                         \
                errors[i] = 0;                                                                               \
                for (size_t j = 0; j < isize; j++) {                                                         \
                    gradients[wi + j] += deltas[i] * inputs[j];                                              \
                }                                                                                            \
            }                                                                                                \
        }                                                                                                    \
    }

#define $DENSE_LEARN(dtype)                                                                                  \
    static void learn_##dtype(pulse_layer_t *layer, size_t batch_size)                                       \
    {                                                                                                        \
        struct pulse_layer_dense_internal_s dense = *(struct pulse_layer_dense_internal_s *)layer->internal; \
                                                                                                             \
        dtype *weights   = (dtype *)dense.w;                                                                 \
        dtype *bias      = (dtype *)dense.b;                                                                 \
        dtype *gradients = (dtype *)dense.g;                                                                 \
        dtype *deltas    = (dtype *)dense.d;                                                                 \
        size_t isize     = layer->isize;                                                                     \
        size_t osize     = layer->osize;                                                                     \
        double HYPER     = -dense.lr / batch_size;                                                           \
        for (size_t i = 0, wi = 0; i < osize; i++, wi += isize) {                                            \
            bias[i] += deltas[i] * HYPER;                                                                    \
            deltas[i] = 0;                                                                                   \
            for (size_t j = 0; j < isize; j++) {                                                             \
                weights[wi + j] += gradients[wi + j] * HYPER;                                                \
                gradients[wi + j] = 0;                                                                       \
            }                                                                                                \
        }                                                                                                    \
        if (layer->next) {                                                                                   \
            layer->next->learn(layer->next, batch_size);                                                     \
        }                                                                                                    \
    }

void $pulse_layer_dense_free_function(pulse_layer_t *layer)
{
    struct pulse_layer_dense_internal_s *dense = (struct pulse_layer_dense_internal_s *)layer->internal;
    free(layer->outputs);
    free(layer->errors);
    free(dense->w);
    free(dense->b);
    free(dense->g);
    free(dense->d);
    free(dense);
};

$DENSE_FORWARD(int)
$DENSE_FORWARD(float)
$DENSE_FORWARD(double)
$DENSE_THINK(int)
$DENSE_THINK(float)
$DENSE_THINK(double)
$DENSE_LEARN(int)
$DENSE_LEARN(float)
$DENSE_LEARN(double)

void (*$pulse_layer_dense_forward_functions[])(pulse_layer_t *, const void *const) = {[PULSE_INT] = forward_int, [PULSE_FLOAT] = forward_float, [PULSE_DOUBLE] = forward_double};
void (*$pulse_layer_dense_think_functions[])(pulse_layer_t *)                      = {[PULSE_INT] = think_int, [PULSE_FLOAT] = think_float, [PULSE_DOUBLE] = think_double};
void (*$pulse_layer_dense_learn_functions[])(pulse_layer_t *, size_t)              = {[PULSE_INT] = learn_int, [PULSE_FLOAT] = learn_float, [PULSE_DOUBLE] = learn_double};

pulse_layer_t pulse_dense_layer(pulse_dtype_t dtype, size_t isize, size_t osize, pulse_activation_t activate, double lr)
{
    const size_t DSIZE = pulse_dtype_sizes[dtype];

    struct pulse_layer_dense_internal_s *$dense = malloc(sizeof(struct pulse_layer_dense_internal_s));
    void *outputs                               = malloc(DSIZE * osize);
    void *errors                                = malloc(DSIZE * osize);
    void *weights                               = malloc(DSIZE * osize * isize);
    void *gradients                             = malloc(DSIZE * osize * isize);
    void *biases                                = malloc(DSIZE * osize);
    void *deltas                                = malloc(DSIZE * osize);

    PULSE_DEBUG_ERROR(isize <= 0, "PulseDenseLayer::Create >> Input length can't be negative or zero.");
    PULSE_DEBUG_ERROR(osize <= 0, "PulseDenseLayer::Create >> Output length can't be negative or zero.");
    PULSE_DEBUG_ERROR(activate == NULL, "PulseDenseLayer::Create >> Activation function can't be NULL.");
    PULSE_DEBUG_ERROR($dense == NULL, "PulseDenseLayer::Create >> Heap memory allocation failed.");
    PULSE_DEBUG_ERROR(outputs == NULL, "PulseDenseLayer::Create >> Heap memory allocation failed.");
    PULSE_DEBUG_ERROR(errors == NULL, "PulseDenseLayer::Create >> Heap memory allocation failed.");
    PULSE_DEBUG_ERROR(weights == NULL, "PulseDenseLayer::Create >> Heap memory allocation failed.");
    PULSE_DEBUG_ERROR(biases == NULL, "PulseDenseLayer::Create >> Heap memory allocation failed.");
    PULSE_DEBUG_ERROR(deltas == NULL, "PulseDenseLayer::Create >> Heap memory allocation failed.");

    const struct pulse_layer_dense_internal_s dense = {
        .w        = weights,
        .b        = biases,
        .g        = gradients,
        .d        = deltas,
        .lr       = lr,
        .activate = activate,
    };

    const pulse_layer_t layer = {
        .outputs  = outputs,
        .errors   = errors,
        .prev     = NULL,
        .next     = NULL,
        .forward  = $pulse_layer_dense_forward_functions[dtype],
        .think    = $pulse_layer_dense_think_functions[dtype],
        .learn    = $pulse_layer_dense_learn_functions[dtype],
        .free     = $pulse_layer_dense_free_function,
        .osize    = osize,
        .isize    = isize,
        .internal = $dense,
    };

    memcpy($dense, &dense, sizeof(struct pulse_layer_dense_internal_s));

    double *$weights = (double *)dense.w;
    for (size_t i = 0; i < isize * osize; i++) $weights[i] = (double)rand() / (double)(RAND_MAX)*sqrt(2.0 / (double)(isize + osize));
    return layer;
}

#undef $DENSE_FORWARD
#undef $DENSE_THINK
#undef $DENSE_LEARN
#endif
