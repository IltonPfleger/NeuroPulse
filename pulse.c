#include <debug/debug.h>
#include <pulse.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void *pulse_forward(struct pulse_layer_s *layers, const void *const inputs)
{
    layers->forward(layers, inputs);
    while (layers->next != NULL) layers = layers->next;
    return layers->outputs;
}

void pulse_think(struct pulse_layer_s *layers)
{
    while (layers->next != NULL) layers = layers->next;
    layers->think(layers);
}

void pulse_learn(struct pulse_layer_s *layers, size_t batch_size) { layers->learn(layers, batch_size); }

void pulse_shuffle(size_t *indexes, size_t max)
{
    for (size_t i = max - 1; i > 0; i--) {
        size_t j   = (size_t)rand() % (i + 1);
        size_t tmp = indexes[i];
        indexes[i] = indexes[j];
        indexes[j] = tmp;
    }
}

void pulse_train(struct pulse_layer_s *layers, size_t epoch, size_t samples, size_t batch_size, pulse_loss_t loss, const void *const *x, const void *const *y)
{
    srand(time(NULL));

    struct pulse_layer_s *output = layers;
    while (output->next != NULL) output = output->next;

    double average_loss = 0;

    size_t RANDOM[samples];
    for (size_t i = 0; i < samples; i++) RANDOM[i] = i;

    for (size_t i = 0; i < epoch; i++) {
        pulse_shuffle(RANDOM, samples);
        for (size_t j = 0; j < samples; j++) {
            pulse_forward(layers, x[RANDOM[j]]);
            average_loss = loss(output->outputs, y[RANDOM[j]], output->errors, output->osize);
            pulse_think(layers);

            if ((j + 1) % batch_size == 0) pulse_learn(layers, batch_size);

            PULSE_DEBUG_LOGGER("Epoch: %ld | Item: %ld | Avg Loss: %.10f\r", i, j, average_loss);
        }
        PULSE_DEBUG_LOGGER("\n");
    }
}

struct pulse_layer_s *pulse_create_model(int size, ...)
{
    va_list args;
    va_start(args, size);
    PULSE_DEBUG_ERROR(size <= 0, "PulseModel::Create >> Size can't be zero.");

    struct pulse_layer_s *layers = malloc(sizeof(struct pulse_layer_s) * size);
    PULSE_DEBUG_ERROR(layers == NULL, "PulseModel::Create >> Heap memory allocation failed.");

    for (size_t i = 0; i < size; i++) {
        struct pulse_layer_s layer = va_arg(args, struct pulse_layer_s);
        layers[i]                  = layer;
        if (i > 0) {
            layers[i - 1].next = &layers[i];
            layers[i].prev     = &layers[i - 1];
        }
    }
    va_end(args);
    return layers;
}

void pulse_free(struct pulse_layer_s *layers)
{
    struct pulse_layer_s *layer = layers;
    while (layer != NULL) {
        layer->free(layer);
        layer = layer->next;
    }
    free(layers);
}
