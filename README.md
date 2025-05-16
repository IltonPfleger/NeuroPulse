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
* [x] Custom Activation Functions Layer.
* [ ] Convolutional Layers.
* [ ] RNN Features.
* [ ] Optimizers like RMSProp, Adam, etc.
* [x] Custom Loss Functions.

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
    constexpr int SAMPLES          = 4;
    constexpr int INPUT_DIMENSION  = 2;
    constexpr int OUTPUT_DIMENSION = 1;

    auto constexpr DTYPE = PULSE_DOUBLE;
    auto const ReLU      = PULSE_RELU[DTYPE];
    auto const Sigmoid   = PULSE_SIGMOID[DTYPE];
    auto const MSE       = PULSE_MSE[DTYPE];

    const double x[SAMPLES][INPUT_DIMENSION]  = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    const double y[SAMPLES][OUTPUT_DIMENSION] = {{1}, {0}, {1}, {0}};
    const void* const X[SAMPLES]              = {x[0], x[1], x[2], x[3]};
    const void* const Y[SAMPLES]              = {y[0], y[1], y[2], y[3]};

    struct pulse_layer_s* model = pulse_create_model(2, pulse_dense_layer(INPUT_DIMENSION, 4, DTYPE, ReLU), pulse_dense_layer(4, OUTPUT_DIMENSION, DTYPE, Sigmoid));

    pulse_train(model, (pulse_train_args_t){.samples = SAMPLES, .epoch = 10000, .batch_size = 1, .lr = 0.1}, MSE, X, Y);

    printf("TRAIN RESULT\n");
    for (int i = 0; i < 4; i++)
        printf("Entrada: %d %d, Output: %f\n", (int)x[i][0], (int)x[i][1], *(double*)pulse_forward(model, x[i]));

    pulse_free(model);
}
```
## Notes:
* The project requires C standard libraries. If using non-compiled files, include them in your compilation.

* Feel free to send ideas, suggestions, questions, and requests.
