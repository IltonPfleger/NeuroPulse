#include <activations/relu.h>
#include <activations/sigmoid.h>
#include <layers/dense.h>
#include <losses/l1.h>
#include <omp.h>
#include <pulse.h>
#include <stdio.h>
#include <stdlib.h>

size_t IMAGE_SIZE  = 784;
size_t OUTPUT_SIZE = 10;
size_t SAMPLES     = 60000;

void **get_train_images()
{
    void **images = (void **)malloc(sizeof(void *) * SAMPLES);
    FILE *file    = fopen("train-images-idx3-ubyte", "rb");
    assert(file != NULL);
    fseek(file, sizeof(int) * 4, SEEK_SET);
    unsigned char buffer[IMAGE_SIZE];
    for (int i = 0; i < SAMPLES; i++) {
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
    void **labels = (void **)malloc(sizeof(void *) * SAMPLES);
    FILE *file    = fopen("train-labels-idx1-ubyte", "rb");
    assert(labels != NULL);
    assert(file != NULL);
    fseek(file, sizeof(int) * 2, SEEK_SET);
    unsigned char value;
    for (int i = 0; i < SAMPLES; i++) {
        labels[i]    = malloc(sizeof(double) * OUTPUT_SIZE);
        double *data = (double *)labels[i];
        fread(&value, 1, 1, file);
        for (int j = 0; j < OUTPUT_SIZE; j++) {
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
    auto DTYPE   = PULSE_DOUBLE;
    auto ReLU    = PULSE_RELU[DTYPE];
    auto SIGMOID = PULSE_SIGMOID[DTYPE];
    auto L1      = PULSE_L1[DTYPE];

    void **images = get_train_images();
    void **labels = get_train_labels();

    print_image((double *)images[0]);
    print_one_hot_label((double *)labels[0]);

    pulse_model model = pulse_create_model(2, pulse_dense_layer(IMAGE_SIZE, 128, DTYPE, ReLU), pulse_dense_layer(128, 10, DTYPE, SIGMOID));

    double t1 = omp_get_wtime();
    pulse_train(model, (pulse_train_args_t){.samples = 60000, .epoch = 10, .batch_size = 100, .lr = 0.5}, L1, images, labels);
    double t2 = omp_get_wtime();
    printf("%f\n", t2 - t1);

    print_one_hot_label((double *)pulse_forward(model, images[0]));

    pulse_free(model);
};
;
