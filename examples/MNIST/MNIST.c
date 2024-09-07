#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../../include/pulse.h"
#define IMAGE_SIZE 784

void * get_train_images() {
    PULSE_DATA * images = (PULSE_DATA*)malloc(sizeof(PULSE_DATA)*IMAGE_SIZE*60000); //PULSE_DATA -> Default -> Single Precision Float
    FILE *fptr;
    fptr = fopen("train-images-idx3-ubyte", "rb");
    fread(images, sizeof(int), 4,  fptr); //TRASH HEADER
    int i = 0;
    unsigned char buffer[IMAGE_SIZE];
    while(fread(buffer, IMAGE_SIZE, 1, fptr)) {
        for(int j = 0; j < IMAGE_SIZE; j++)
            images[i + j] = buffer[j] > 0 ? 1.f:0.f;
        i += 784;
    };
    return images;
}

void print_image(PULSE_DATA * image) {
    for(int i = 0; i < IMAGE_SIZE; i++)
        if(i%28 != 0)
            printf("%.f", image[i]);
        else
            printf("\n");
    printf("\n");
}

void * get_train_labels() {
    const int SIZE = 60000*10;
    PULSE_DATA * labels = (PULSE_DATA*)malloc(sizeof(PULSE_DATA)*SIZE);
    FILE *fptr;
    fptr = fopen("train-labels-idx1-ubyte", "rb");
    fread(labels, sizeof(int), 2,  fptr); //TRASH HEADER
    int i = 0;
    unsigned char dst;
    while(fread(&dst, 1, 1, fptr)) {
        for(int j = 0; j < 10; j++)
            labels[i + j] = j == dst ? 1:0;
        i += 10;
    }
    return labels;
}

void print_one_hot_label(PULSE_DATA * label) {
    printf("[ ");
    for(int i = 0; i < 10; i++)
        printf("%.f ", label[i]);
    printf("]\n");
}




int main() {
    PULSE_DATA * images = get_train_images();
    PULSE_DATA * labels = get_train_labels();

    print_image(images);
    print_one_hot_label(labels);

    pulse_model model = pulse_create_model(2,
                                           pulse_create_dense_layer(IMAGE_SIZE, 128, PULSE_ACTIVATION_RELU, PULSE_OPTIMIZATION_SIMD),
                                           pulse_create_dense_layer(128, 10, PULSE_ACTIVATION_SIGMOID, PULSE_OPTIMIZATION_SIMD));


    double t1 = omp_get_wtime();
    pulse_train(model, 5, 60000, (pulse_train_hyper_args_t) {
        100, 0.1
    }, PULSE_LOSS_MSE, (PULSE_DATA*)images, (PULSE_DATA*)labels);
    double t2 = omp_get_wtime();
    printf("%f\n", t2 - t1);
    print_one_hot_label(pulse_foward(model.layers, images));
};
