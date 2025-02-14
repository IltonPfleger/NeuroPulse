# NeuroPulse

![C](https://badgen.net/badge/MADE-WITH/C)
![LICENSE](https://badgen.net/badge/LICENSE/MIT/green)
![ML](https://badgen.net/badge/ML/Machine-Learning/red)
![PULSE](https://badgen.net/badge/Learn%20in%20a/PULSE⚡/yellow)


<div align="center">
<img width="30%" src="https://github.com/IltonPfleger/NeuroPulse/blob/main/preview.gif">
</div>

# About The Project

The aim of this project is to create a user-friendly Neural Networks library for the C programming language. The library should be easy for users to understand and modify. The core concept is to make everything modular, enabling users to adapt architectures to solve their problems. 

## Features
* [x] Stochastic Gradient Descent.
* [x] Batch Gradient Descent.
* [x] Mini-Batch Gradient Descent.
* [x] Custom Activation Functions Per Layer.
* [x] Convolutional Layers.
* [ ] RNN Features.
* [ ] Optimizers like RMSProp, Adam, etc.
* [x] Custom Error Functions.

## Example
```c Xor Problem.
#include <activations/relu.h>
#include <activations/sigmoid.h>
#include <layers/dense.h>
#include <losses/mse.h>
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
    pulse_train(model, (pulse_train_args_t){.samples = 4, .epoch = 15000, .batch_size = 1, .lr = 0.1}, MSE, X, Y);

    printf("TRAIN RESULT\n");
    for (int i = 0; i < 4; i++) {
        double* output = pulse_forward(model, x[i]);
        printf("Entrada: %d %d, Output: %f\n", (int)x[i][0], (int)x[i][1], *output);
    }

    pulse_free(model);
}
```
## Notes:
* The project requires C standard libraries. If using non-compiled files, include them in your compilation.

* Feel free to send ideas, suggestions, questions, and requests.
