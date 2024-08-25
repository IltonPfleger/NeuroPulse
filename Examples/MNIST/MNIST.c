#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../../Include/PULSE.h"
#define IMAGE_SIZE 784

void * get_train_images()
{
    PULSE_data_t * images = (PULSE_data_t*)malloc(sizeof(PULSE_data_t)*IMAGE_SIZE*60000); //PULSE_data_t -> Default -> Single Precision Float
    FILE *fptr;
    fptr = fopen("train-images-idx3-ubyte", "rb");
    fread(images, sizeof(int), 4,  fptr); //TRASH HEADER
    int i = 0;
    unsigned char buffer[IMAGE_SIZE];
    while(fread(buffer, IMAGE_SIZE, 1, fptr))
    {
        for(int j = 0; j < IMAGE_SIZE; j++)
            images[i + j] = buffer[j] > 0 ? 1.f:0.f;
        i += 784;
    };
    return images;
}

void print_image(PULSE_data_t * image)
{
    for(int i = 0; i < IMAGE_SIZE; i++)
        if(i%28 != 0)
            printf("%.f", image[i]);
        else
            printf("\n");
    printf("\n");
}

void * get_train_labels()
{
    const int SIZE = 60000*10;
    PULSE_data_t * labels = (PULSE_data_t*)malloc(sizeof(PULSE_data_t)*SIZE);
    FILE *fptr;
    fptr = fopen("train-labels-idx1-ubyte", "rb");
    fread(labels, sizeof(int), 2,  fptr); //TRASH HEADER
    int i = 0;
    unsigned char dst;
    while(fread(&dst, 1, 1, fptr))
    {
        for(int j = 0; j < 10; j++)
            labels[i + j] = j == dst ? 1:0;
        i += 10;
    }
    return labels;
}

void print_one_hot_label(PULSE_data_t * label)
{
    printf("[ ");
    for(int i = 0; i < 10; i++)
        printf("%.f ", label[i]);
    printf("]\n");
}




int main()
{
    PULSE_data_t * images = get_train_images();
    PULSE_data_t * labels = get_train_labels();

    print_image(images);
    print_one_hot_label(labels);

    PULSE_Model model = PULSE_CreateModel(2,
    PULSE_DENSE, (PULSE_DenseLayerArgs) {
        IMAGE_SIZE, 128, PULSE_ACTIVATION_RELU, PULSE_OPTIMIZATION_SIMD
    },
    PULSE_DENSE,(PULSE_DenseLayerArgs) {
        128, 10, PULSE_ACTIVATION_SIGMOID, PULSE_OPTIMIZATION_SIMD
    });


    double t1 = omp_get_wtime();
    PULSE_Train(model, 5, 60000, (PULSE_HyperArgs) {
        100, 0.1
    }, PULSE_LOSS_MSE, (PULSE_data_t*)images, (PULSE_data_t*)labels);
    double t2 = omp_get_wtime();
    printf("%f\n", t2 - t1);
    print_one_hot_label(PULSE_Foward(model.layers, images));
};
