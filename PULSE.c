#include "Include/PULSE.h"


PULSE_DataType * PULSE_Foward(PULSE_Layer * layer, PULSE_DataType * inputs)
{
    if(inputs != NULL)
        memcpy(layer->inputs, inputs, sizeof(PULSE_DataType)*layer->n_inputs);
    layer->feed(layer);

    if(layer->child != NULL)
    {
        memcpy(layer->child->inputs, layer->outputs, sizeof(PULSE_DataType)*layer->n_outputs);
        return PULSE_Foward(layer->child, NULL);
    }
    else
        return layer->outputs;
}


void PULSE_Back(PULSE_Layer * layer)
{
    layer->back(layer);
    if(layer->parent != NULL)
    {
        PULSE_Back(layer->parent);
        memset(layer->parent->errors, 0, layer->n_inputs*sizeof(PULSE_DataType));
    }
}

void PULSE_Fix(PULSE_DataType * weights, PULSE_DataType * fixes, PULSE_HyperArgs args, size_t size)
{
    const float HYPER = -args.lr/args.batch_size;
    for(int i = 0; i < size; i++)
    {
        weights[i] += fixes[i] * HYPER;
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

void PULSE_Train(PULSE_Model model, size_t epoch, size_t data_size, PULSE_HyperArgs args, PULSE_LossFunction loss_function, PULSE_DataType * x, PULSE_DataType * y)
{
    srand(time(NULL));
    PULSE_LossFunctionPtr PULSE_GetLoss = PULSE_GetLossFunctionPtr(loss_function);
    PULSE_DataType * FIXES = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*model.fixes_size);
    PULSE_DataType * FIXES_PTR = FIXES;
    PULSE_DataType * ERRORS = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*model.errors_size);
    PULSE_DataType * ERRORS_PTR = ERRORS;

    PULSE_Layer * output = model.layers + model.n_layers - 1;
    PULSE_DataType loss = 0, batch_loss = 0;

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

void PULSE_Connect(PULSE_Layer * parent, PULSE_Layer * child)
{
    parent->child = child;
    child->parent = parent;
}

PULSE_Model PULSE_CreateModel(int size, ...)
{
    size *= 2;
    PULSE_Layer * layers = (PULSE_Layer*)malloc(sizeof(PULSE_Layer)*size/2);

    va_list layers_info;
    va_start(layers_info, size);
    unsigned int WEIGHTS_SIZE = 0;
    unsigned int IO_SIZE = 0;
    unsigned int FIXES_SIZE = 0;

    for (int i = 0; i < size/2; i++)
    {
        PULSE_LayerType type = va_arg(layers_info, PULSE_LayerType);
        switch(type)
        {
        case PULSE_DENSE:
            PULSE_DenseLayerArgs args = va_arg(layers_info, PULSE_DenseLayerArgs);
            WEIGHTS_SIZE += PULSE_GetDenseWeightsSize(args);
            IO_SIZE += PULSE_GetDenseIOSize(args);
            FIXES_SIZE += PULSE_GetDenseFixesSize(args);
            break;
        }
    }

    va_end(layers_info);
    va_start(layers_info, size);
    PULSE_DataType * WEIGHTS = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*WEIGHTS_SIZE);
    PULSE_DataType * IO = (PULSE_DataType*)aligned_alloc(__PULSE_CFLAGS_CacheLineSize, sizeof(PULSE_DataType)*IO_SIZE);
    PULSE_DataType * WEIGHTS_PTR = WEIGHTS;
    PULSE_DataType * IO_PTR = IO;

    for (int i = 0; i < size/2; i++)
    {
        PULSE_LayerType type = va_arg(layers_info, PULSE_LayerType);
        switch(type)
        {
        case PULSE_DENSE:
            PULSE_DenseLayerArgs args = va_arg(layers_info, PULSE_DenseLayerArgs);
            layers[i] = PULSE_CreateDenseLayer(args, WEIGHTS_PTR, IO_PTR);
            WEIGHTS_PTR += PULSE_GetDenseWeightsSize(args);
            IO_PTR += PULSE_GetDenseIOSize(args);
            break;
        }
        if(i > 0)
            PULSE_Connect(&layers[i - 1], &layers[i]);
    }
    va_end(layers_info);
    return (PULSE_Model) {
        layers, WEIGHTS, IO, size/2, WEIGHTS_SIZE, IO_SIZE, FIXES_SIZE, 129
    };
}


void PULSE_Destroy(PULSE_Model * model)
{
    free(model->weights);
    free(model->layers);
    free(model->io);
}
