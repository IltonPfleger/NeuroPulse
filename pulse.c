#include "pulse.h"

PULSE_DATA * pulse_foward(pulse_layer_t * layer, PULSE_DATA * inputs) {
    if(inputs != NULL) memcpy(layer->inputs, inputs, sizeof(PULSE_DATA)*layer->n_inputs);
    layer->feed(layer);
    if(layer->next != NULL) return pulse_foward(layer->next, NULL);
    else return layer->outputs;
}


void pulse_back(pulse_layer_t * layer) {
    layer->back(layer);
    if(layer->prev != NULL) {
        pulse_back(layer->prev);
        memset(layer->prev->errors, 0, layer->n_inputs*sizeof(PULSE_DATA));
    }
}

void pulse_fix(PULSE_DATA * restrict weights, PULSE_DATA * restrict fixes, pulse_train_hyper_args_t args, size_t size) {
    const float HYPER = -args.lr/args.batch_size;
    for(int i = 0; i < size; i ++) {
        weights[i] += HYPER * fixes[i];
        fixes[i] = 0;
    }
}



void pulse_shuffle(size_t *indexes, size_t max) {
    for (int i = 0; i < max; i++) {
        size_t random = (size_t)rand() % max;
        size_t random2 = (size_t)rand() % max;
        size_t temp = indexes[random];
        indexes[random] = indexes[random2];
        indexes[random2] = temp;
    };
}

void pulse_train(pulse_model model, size_t epoch, size_t data_size, pulse_train_hyper_args_t args, pulse_loss_fnc_e loss_function, PULSE_DATA * x, PULSE_DATA * y) {
    srand(time(NULL));
    pulse_loss_fnc_ptr get_loss = pulse_get_loss_fnc_ptr(loss_function);
    PULSE_DATA * FIXES = (PULSE_DATA*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DATA)*model.fixes_size);
    PULSE_DATA * FIXES_PTR = FIXES;
    PULSE_DATA * ERRORS = (PULSE_DATA*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DATA)*model.errors_size);
    PULSE_DATA * ERRORS_PTR = ERRORS;

    pulse_layer_t * output = model.layers + model.n_layers - 1;
    PULSE_DATA loss = 0, batch_loss = 0;

    for(int i = 0; i < model.n_layers; i++)
        model.layers[i].mode(model.layers + i, &FIXES_PTR, &ERRORS_PTR);

    size_t RANDOM[data_size];
    for (int i = 0; i < data_size; i++)
        RANDOM[i] = i;

    for (int i = 0; i < epoch; i++) {
        pulse_shuffle(RANDOM, data_size);
        for (int j = 0; j < data_size; j++) {
            pulse_foward(model.layers, x + RANDOM[j]*model.layers->n_inputs);
            loss = get_loss(output->outputs, y + RANDOM[j]*output->n_outputs, output->errors, output->n_outputs);
            pulse_back(output);

            if((j+1)%args.batch_size == 0)
                pulse_fix(model.weights, FIXES, args, model.weights_size);

            batch_loss += loss/data_size;
            printf("Epoch: %d | Item: %d | Loss: %.10f | Batch Loss: %.10f\r", i, j, loss, batch_loss);
        }
        batch_loss = 0;
    }
    free(FIXES);
    free(ERRORS);
}

pulse_model pulse_create_model(int size, ...) {
    srand(time(NULL));
    pulse_model model;
    model.n_layers = size;
    model.layers = (pulse_layer_t*)malloc(sizeof(pulse_layer_t)*model.n_layers);

    va_list layers_info;
    va_start(layers_info, size);
    model.weights_size = 0;
    model.io_size = 0;
    model.fixes_size = 0;
    model.errors_size = 0;

    for(int i = 0; i < model.n_layers; i++) {
        pulse_layer_t layer = va_arg(layers_info, pulse_layer_t);
        model.io_size += layer.n_inputs;
        model.errors_size += layer.n_outputs;
        model.weights_size += layer.get_weights_size(&layer);
        model.fixes_size += layer.get_weights_size(&layer);
        model.layers[i] = layer;
        if(i == model.n_layers - 1) model.io_size += layer.n_outputs;
        if(i > 0) {
            model.layers[i - 1].next = &model.layers[i];
            model.layers[i].prev = &model.layers[i - 1];
        }
    }

    model.weights = (PULSE_DATA*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DATA)*model.weights_size);
    model.io = (PULSE_DATA*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DATA)*model.io_size);
    PULSE_DATA * WEIGHTS_PTR = model.weights, * IO = model.io;
    pulse_layer_t * current = model.layers;
    while(current != NULL) {
        current->start(current, &WEIGHTS_PTR, &IO);
        current->randomize(current);
        current = current->next;
    }

    va_end(layers_info);
    return model;
}


void pulse_destroy(pulse_model * model) {
    free(model->weights);
    free(model->layers);
    free(model->io);
}
