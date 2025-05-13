#include <activations/relu.h>
#include <activations/sigmoid.h>
#include <layers/dense.h>
#include <losses/l1.h>
#include <pulse.h>
#include <stdio.h>
#include <stdlib.h>

constexpr int IMAGE_SIZE      = 784;
constexpr int N_TRAIN_SAMPLES = 60000;

void **get_train_images()
{
    void **images = (void **)malloc(sizeof(void *) * N_TRAIN_SAMPLES);
    FILE *file    = fopen("train-images-idx3-ubyte", "rb");
    assert(file != NULL);
    fseek(file, sizeof(int) * 4, SEEK_SET);
    unsigned char buffer[IMAGE_SIZE];
    for (int i = 0; i < N_TRAIN_SAMPLES; i++) {
        images[i]    = malloc(sizeof(double) * IMAGE_SIZE);
        double *data = (double *)images[i];
        fread(buffer, IMAGE_SIZE, 1, file);
        for (int j = 0; j < IMAGE_SIZE; j++) {
            data[j] = buffer[j] > 0 ? 1.f : 0.f;
        }
    }
    return images;
}

void print_image(double *image)
{
    for (int i = 0; i < IMAGE_SIZE; i++) {
        if (i % 28 != 0) {
            printf("%.f", image[i]);
        } else {
            printf("\n");
        }
    }
    printf("\n");
}
//
void **get_train_labels()
{
    void **labels = (void **)malloc(sizeof(void *) * N_TRAIN_SAMPLES);
    FILE *file    = fopen("train-labels-idx1-ubyte", "rb");
    assert(labels != NULL);
    assert(file != NULL);
    fseek(file, sizeof(int) * 2, SEEK_SET);
    unsigned char value;
    for (int i = 0; i < N_TRAIN_SAMPLES; i++) {
        labels[i]    = malloc(sizeof(double) * 10);
        double *data = (double *)labels[i];
        fread(&value, 1, 1, file);
        for (int j = 0; j < 10; j++) {
            data[j] = j == value ? 1 : 0;
        }
    }
    return labels;
}

void print_one_hot_label(double *label)
{
    printf("[ ");
    for (int i = 0; i < 10; i++) printf("%.f ", label[i]);
    printf("]\n");
}

int main()
{
    constexpr int SAMPLES          = N_TRAIN_SAMPLES;
    constexpr int INPUT_DIMENSION  = IMAGE_SIZE;
    constexpr int OUTPUT_DIMENSION = 10;
    constexpr int BATCH_SIZE       = 100;
    constexpr int EPOCH            = 10;
    constexpr double LR            = 0.1;

    auto constexpr DTYPE   = PULSE_DOUBLE;
    auto constexpr ReLU    = PULSE_RELU[DTYPE];
    auto constexpr SIGMOID = PULSE_SIGMOID[DTYPE];
    auto constexpr L1      = PULSE_L1[DTYPE];

    auto images = (const void *const *)get_train_images();
    auto labels = (const void *const *)get_train_labels();

    pulse_model model = pulse_create_model(2, pulse_dense_layer(INPUT_DIMENSION, 128, DTYPE, ReLU), pulse_dense_layer(128, OUTPUT_DIMENSION, DTYPE, SIGMOID));

    clock_t t1 = clock();
    pulse_train(model, (pulse_train_args_t){.samples = SAMPLES, .epoch = EPOCH, .batch_size = BATCH_SIZE, .lr = LR}, L1, images, labels);
    clock_t t2 = clock();
    printf("%f\n", (double)((t2 - t1) / (double)(CLOCKS_PER_SEC)));

    pulse_free(model);
};
;
