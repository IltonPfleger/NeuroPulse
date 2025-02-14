#include <activations/relu.h>
#include <activations/sigmoid.h>
#include <layers/dense.h>
#include <losses/mse.h>
#include <omp.h>
#include <pulse.h>
#include <stdio.h>

int main()
{
    auto DTYPE   = PULSE_DOUBLE;
    auto ReLU    = PULSE_RELU[DTYPE];
    auto Sigmoid = PULSE_SIGMOID[DTYPE];
    auto MSE     = PULSE_MSE[DTYPE];

    double x[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    double y[4][1] = {{1}, {0}, {1}, {0}};
    void* X[4];
    void* Y[4];

    X[0] = x[0];
    X[1] = x[1];
    X[2] = x[2];
    X[3] = x[3];

    Y[0] = y[0];
    Y[1] = y[1];
    Y[2] = y[2];
    Y[3] = y[3];

    pulse_model model = pulse_create_model(2, pulse_dense_layer(2, 4, DTYPE, ReLU), pulse_dense_layer(4, 1, DTYPE, Sigmoid));

    double t1 = omp_get_wtime();
    pulse_train(model, (pulse_train_args_t){.samples = 4, .epoch = 15000, .batch_size = 1, .lr = 0.1}, MSE, X, Y);
    double t2 = omp_get_wtime();
    printf("%f\n", t2 - t1);

    printf("TRAIN RESULT\n");
    for (int i = 0; i < 4; i++) {
        double* output = pulse_forward(model, x[i]);
        printf("Entrada: %d %d, Output: %f\n", (int)x[i][0], (int)x[i][1], *output);
    }

    pulse_free(model);
}
