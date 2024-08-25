#include "Include/PULSE.h"

PULSE_data_t * PULSE_Foward(PULSE_layer_t * layer, PULSE_data_t * inputs)
{
    if(inputs != NULL)
        memcpy(layer->inputs, inputs, sizeof(PULSE_data_t)*layer->n_inputs);
    layer->feed(layer);

    if(layer->child != NULL)
    {
        memcpy(layer->child->inputs, layer->outputs, sizeof(PULSE_data_t)*layer->n_outputs);
        return PULSE_Foward(layer->child, NULL);
    }
    else
        return layer->outputs;
}


void PULSE_Back(PULSE_layer_t * layer)
{
    layer->back(layer);
    if(layer->parent != NULL)
    {
        PULSE_Back(layer->parent);
        memset(layer->parent->errors, 0, layer->n_inputs*sizeof(PULSE_data_t));
    }
}

void PULSE_Fix(PULSE_data_t * restrict weights, PULSE_data_t * restrict fixes, PULSE_HyperArgs args, size_t size)
{
    const float HYPER = -args.lr/args.batch_size;
    for(int i = 0; i < size; i ++)
    {
        weights[i] += HYPER * fixes[i];
        fixes[i] = 0;
    }
}



void PULSE_Shuffle(size_t *indexes, size_t max)
{
    for (int i = 0; i < max; i++)
    {
        size_t random = (size_t)rand() % max;
        size_t random2 = (size_t)rand() % max;
        size_t temp = indexes[random];
        indexes[random] = indexes[random2];
        indexes[random2] = temp;
    };
}

void PULSE_Train(PULSE_Model model, size_t epoch, size_t data_size, PULSE_HyperArgs args, PULSE_LossFunction loss_function, PULSE_data_t * x, PULSE_data_t * y)
{
    srand(time(NULL));
    PULSE_LossFunctionPtr PULSE_GetLoss = PULSE_GetLossFunctionPtr(loss_function);
    PULSE_data_t * FIXES = (PULSE_data_t*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_data_t)*model.fixes_size);
    PULSE_data_t * FIXES_PTR = FIXES;
    PULSE_data_t * ERRORS = (PULSE_data_t*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_data_t)*model.errors_size);
    PULSE_data_t * ERRORS_PTR = ERRORS;

    PULSE_layer_t * output = model.layers + model.n_layers - 1;
    PULSE_data_t loss = 0, batch_loss = 0;

    for(int i = 0; i < model.n_layers; i++)
        model.layers[i].mode(model.layers + i, &FIXES_PTR, &ERRORS_PTR);

    size_t RANDOM[data_size];
    for (int i = 0; i < data_size; i++)
        RANDOM[i] = i;

    for (int i = 0; i < epoch; i++)
    {
        PULSE_Shuffle(RANDOM, data_size);
        for (int j = 0; j < data_size; j++)
        {
            PULSE_Foward(model.layers, x + RANDOM[j]*model.layers->n_inputs);
            loss = PULSE_GetLoss(output->outputs, y + RANDOM[j]*output->n_outputs, output->errors, output->n_outputs);
            PULSE_Back(output);

            if((j+1)%args.batch_size == 0)
                PULSE_Fix(model.weights, FIXES, args, model.weights_size);

            batch_loss += loss/data_size;
            printf("Epoch: %d | Item: %d | Loss: %.10f | Batch Loss: %.10f\r", i, j, loss, batch_loss);
        }
        batch_loss = 0;
    }
    free(FIXES);
    free(ERRORS);
}

void PULSE_Connect(PULSE_layer_t * parent, PULSE_layer_t * child)
{
    parent->child = child;
    child->parent = parent;
}


PULSE_Model PULSE_CreateModel(int size, ...)
{
    srand(time(NULL));
    PULSE_Model model;
    model.n_layers = size;
    model.layers = (PULSE_layer_t*)malloc(sizeof(PULSE_layer_t)*model.n_layers);

    va_list layers_info;
    va_start(layers_info, size);
    model.weights_size = 0;
    model.io_size = 0;
    model.fixes_size = 0;
    model.errors_size = 0;
    model.trained = 0;

    for(int i = 0; i < model.n_layers; i++)
    {
        PULSE_layer_enum_t type = va_arg(layers_info, PULSE_layer_enum_t);
        switch(type)
        {
        case PULSE_DENSE:
            PULSE_DenseLayerArgs args = va_arg(layers_info, PULSE_DenseLayerArgs);
            model.layers[i] = PULSE_CreateDenseLayer(args);
            model.io_size += model.layers[i].n_outputs + model.layers[i].n_inputs;
            model.errors_size += model.layers[i].n_outputs;
            model.weights_size += model.layers[i].n_inputs * model.layers[i].n_outputs + model.layers[i].n_outputs;
            model.fixes_size += model.layers[i].n_inputs * model.layers[i].n_outputs + model.layers[i].n_outputs;
        }
        if(i > 0)
            PULSE_Connect(&model.layers[i - 1], &model.layers[i]);

    }

    model.weights = (PULSE_data_t*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_data_t)*model.weights_size);
    model.io = (PULSE_data_t*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_data_t)*model.io_size);
    PULSE_data_t * WEIGHTS_PTR = model.weights, * IO = model.io;
    PULSE_layer_t * current = model.layers;
    while(current != NULL)
    {
        current->start(current, &WEIGHTS_PTR, &IO);
        current->randomize(current);
        current = current->child;
    }

    va_end(layers_info);
    return model;
}


void PULSE_Destroy(PULSE_Model * model)
{
    free(model->weights);
    free(model->layers);
    free(model->io);
}
