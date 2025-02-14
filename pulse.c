#include <memory/memory.h>
#include <pulse.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

void *pulse_forward(pulse_model model, void *inputs)
{
    model.layers->feed(model.layers, inputs);
    return (model.layers + model.n_layers - 1)->outputs;
}

void pulse_back(pulse_model model) { (model.layers + model.n_layers - 1)->back(model.layers + model.n_layers - 1); }

void pulse_fix(pulse_model model, pulse_train_args_t args) { model.layers->fix(model.layers, args); }

void pulse_shuffle(size_t *indexes, size_t max)
{
    for (size_t i = 0; i < max; i++) {
        size_t random    = (size_t)rand() % max;
        size_t random2   = (size_t)rand() % max;
        size_t temp      = indexes[random];
        indexes[random]  = indexes[random2];
        indexes[random2] = temp;
    };
}

void pulse_train(pulse_model model, pulse_train_args_t args, pulse_loss_function loss_function, void **x, void **y)
{
    srand(time(NULL));
    pulse_layer_t *output = model.layers + model.n_layers - 1;
    double loss           = 0;

    size_t RANDOM[args.samples];
    for (size_t i = 0; i < args.samples; i++) RANDOM[i] = i;

    for (size_t i = 0; i < args.epoch; i++) {
        pulse_shuffle(RANDOM, args.samples);
        for (size_t j = 0; j < args.samples; j++) {
            pulse_forward(model, x[RANDOM[j]]);
            loss = loss_function(output->outputs, y[RANDOM[j]], output->errors, output->osize);
            pulse_back(model);

            if ((j + 1) % args.batch_size == 0) pulse_fix(model, args);

            printf("Epoch: %ld | Item: %ld | Avg Loss: %.10f\r", i, j, loss);
        }
    }
}

pulse_model pulse_create_model(int size, ...)
{
    srand(time(NULL));
    pulse_model model;
    model.n_layers = size;
    model.layers   = pulse_memory_alloc(sizeof(pulse_layer_t) * model.n_layers);

    va_list layers_info;
    va_start(layers_info, size);

    for (size_t i = 0; i < model.n_layers; i++) {
        pulse_layer_t layer = va_arg(layers_info, pulse_layer_t);
        model.layers[i]     = layer;
        if (i > 0) {
            model.layers[i - 1].next = &model.layers[i];
            model.layers[i].prev     = &model.layers[i - 1];
        }
    }
    va_end(layers_info);
    return model;
}

void pulse_free(pulse_model model)
{
    for (size_t i = 0; i < model.n_layers; i++) model.layers[i].free(model.layers + i);
    pulse_memory_free(model.layers);
}
