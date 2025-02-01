#include <pulse.h>

PULSE_DATA *pulse_foward(pulse_layer_t *layer, PULSE_DATA *inputs) {
    if (inputs != NULL) memcpy(layer->inputs, inputs, sizeof(PULSE_DATA) * layer->n_inputs);
    layer->feed(layer);
    if (layer->next != NULL) {
        memcpy(layer->next->inputs, layer->outputs, sizeof(PULSE_DATA) * layer->n_outputs);
        return pulse_foward(layer->next, NULL);
    } else
        return layer->outputs;
}

void pulse_back(pulse_layer_t *layer) {
    layer->back(layer);
    if (layer->prev != NULL) {
        pulse_back(layer->prev);
        memset(layer->prev->errors, 0, layer->n_inputs * sizeof(PULSE_DATA));
    }
}

void pulse_fix(pulse_model model, pulse_train_hyper_args_t args) {
    const float HYPER = -args.lr / args.batch_size;
    for (int i = 0; i < model.n_layers; i++) {
        for (int j = 0; j < model.layers[i].n_weights; j++) {
            model.layers[i].w[j] += model.layers[i].g[j] * HYPER;
            model.layers[i].g[j] = 0;
        }
    }
}

void pulse_shuffle(size_t *indexes, size_t max) {
    for (int i = 0; i < max; i++) {
        size_t random    = (size_t)rand() % max;
        size_t random2   = (size_t)rand() % max;
        size_t temp      = indexes[random];
        indexes[random]  = indexes[random2];
        indexes[random2] = temp;
    };
}

void pulse_train(pulse_model model, size_t epoch, size_t data_size, pulse_train_hyper_args_t args, pulse_loss_fnc_e loss_function, PULSE_DATA *x,
                 PULSE_DATA *y) {
    srand(time(NULL));
    pulse_loss_fnc_ptr get_loss = pulse_get_loss_fnc_ptr(loss_function);
    pulse_layer_t *output       = model.layers + model.n_layers - 1;
    PULSE_DATA loss = 0, batch_loss = 0;

    size_t RANDOM[data_size];
    for (int i = 0; i < data_size; i++) RANDOM[i] = i;

    for (int i = 0; i < epoch; i++) {
        pulse_shuffle(RANDOM, data_size);
        for (int j = 0; j < data_size; j++) {
            pulse_foward(model.layers, x + RANDOM[j] * model.layers->n_inputs);
            loss = get_loss(output->outputs, y + RANDOM[j] * output->n_outputs, output->errors, output->n_outputs);
            pulse_back(output);

            if ((j + 1) % args.batch_size == 0) pulse_fix(model, args);

            batch_loss += loss / data_size;
            printf("Epoch: %d | Item: %d | Avg Loss: %.10f | Avg Batch Loss: %.10f\r", i, j, loss, batch_loss);
        }
        batch_loss = 0;
    }
}

pulse_model pulse_create_model(int size, ...) {
    srand(time(NULL));
    pulse_model model;
    model.n_layers = size;
    model.layers   = (pulse_layer_t *)malloc(sizeof(pulse_layer_t) * model.n_layers);

    va_list layers_info;
    va_start(layers_info, size);

    for (int i = 0; i < model.n_layers; i++) {
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

void pulse_free(pulse_model *model) {
    for (int i = 0; i < model->n_layers; i++) model->layers[i].free(model->layers + i);
}
