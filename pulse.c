#include <debug/debug.h>
#include <pulse.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void *pulse_forward(struct pulse_layer_s *layers, const void *const inputs)
{
    layers->feed(layers, inputs);
    while (layers->next != NULL) layers = layers->next;
    return layers->outputs;
}

void pulse_back(struct pulse_layer_s *layers)
{
    while (layers->next != NULL) layers = layers->next;
    layers->back(layers);
}

void pulse_fix(struct pulse_layer_s *layers, pulse_train_args_t args) { layers->fix(layers, args); }

void pulse_shuffle(size_t *indexes, size_t max)
{
    for (size_t i = 0; i < max; i++) {
        size_t random    = (size_t)rand() % max;
        size_t random2   = (size_t)rand() % max;
        size_t tmp       = indexes[random];
        indexes[random]  = indexes[random2];
        indexes[random2] = tmp;
    };
}

void pulse_train(struct pulse_layer_s *layers, pulse_train_args_t args, pulse_loss_function loss_function, const void *const *x, const void *const *y)
{
    srand(time(NULL));

    struct pulse_layer_s *output = layers;
    while (output->next != NULL) output = output->next;

    double loss = 0;

    size_t RANDOM[args.samples];
    for (size_t i = 0; i < args.samples; i++) RANDOM[i] = i;

    for (size_t i = 0; i < args.epoch; i++) {
        pulse_shuffle(RANDOM, args.samples);
        for (size_t j = 0; j < args.samples; j++) {
            pulse_forward(layers, x[RANDOM[j]]);
            loss = loss_function(output->outputs, y[RANDOM[j]], output->errors, output->osize);
            pulse_back(layers);

            if ((j + 1) % args.batch_size == 0) pulse_fix(layers, args);

            PULSE_DEBUG_LOGGER("Epoch: %ld | Item: %ld | Avg Loss: %.10f\r", i, j, loss);
        }
        PULSE_DEBUG_LOGGER("\n");
    }
}

struct pulse_layer_s *pulse_create_model(int size, ...)
{
    srand(time(NULL));
    va_list args;
    va_start(args, size);

    struct pulse_layer_s *layers = malloc(sizeof(struct pulse_layer_s) * size);
    PULSE_DEBUG_ERROR(layers == NULL, "PulseModel::Create >> Heap memory allocation failed.");

    for (size_t i = 0; i < size; i++) {
        struct pulse_layer_s layer = va_arg(args, struct pulse_layer_s);
        memcpy(layers + i, &layer, sizeof(struct pulse_layer_s));
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
    while (layers != NULL) {
        layers->free(layers);
        layers = layers->next;
    }
    free(layers);
}
