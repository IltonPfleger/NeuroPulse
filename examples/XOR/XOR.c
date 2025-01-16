#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "../../include/pulse.h"

int main()
{
    PULSE_DATA x[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    PULSE_DATA y[4][1] = {{1}, {0}, {1}, {0}};

    pulse_model model = pulse_create_model(2,
                                           pulse_create_dense_layer(2, 4, PULSE_ACTIVATION_RELU, PULSE_OPTIMIZATION_NONE),
                                           pulse_create_dense_layer(4, 1, PULSE_ACTIVATION_RELU, PULSE_OPTIMIZATION_NONE));
    double t1 = omp_get_wtime();
    pulse_train(model, 15000, 4, (pulse_train_hyper_args_t) {2, 0.1}, PULSE_LOSS_MSE, (PULSE_DATA*)x, (PULSE_DATA*)y);
    double t2 = omp_get_wtime();
    printf("%f\n", t2 - t1);

    printf("TRAIN RESULT\n");
    for (int i = 0; i < 4; i++) {
        printf("Entrada: %d %d, Output: %f\n", (int)x[i][0], (int)x[i][1], pulse_foward(model.layers, x[i])[0]);
    }

    pulse_free(&model);
}
