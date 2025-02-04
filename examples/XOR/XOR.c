#include <activations/activations.h>
#include <layers/dense.h>
#include <omp.h>
#include <pulse.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    pulse_datatype x[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    pulse_datatype y[4][1] = {{1}, {0}, {1}, {0}};

    pulse_model model = pulse_create_model(2, pulse_create_dense_layer(2, 4, pulse_relu), pulse_create_dense_layer(4, 1, pulse_relu));

    double t1 = omp_get_wtime();
    pulse_train(model, 15000, 4, (pulse_train_hyper_args_t){2, 0.1}, pulse_mse, (pulse_datatype*)x, (pulse_datatype*)y);
    double t2 = omp_get_wtime();

    printf("%f\n", t2 - t1);

    printf("TRAIN RESULT\n");
    for (int i = 0; i < 4; i++) {
        printf("Entrada: %d %d, Output: %f\n", (int)x[i][0], (int)x[i][1], pulse_foward(model.layers, x[i])[0]);
    }

    // pulse_free(&model);
}
