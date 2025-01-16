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
#include <stdlib.h>
#include <stdio.h>
#include "../../include/pulse.h"

int main()
{
    PULSE_DATA x[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    PULSE_DATA y[4][1] = {{1}, {0}, {1}, {0}};

    pulse_model model = pulse_create_model(2,
                                           pulse_create_dense_layer(2, 4, PULSE_ACTIVATION_RELU, PULSE_OPTIMIZATION_NONE),
                                           pulse_create_dense_layer(4, 1, PULSE_ACTIVATION_RELU, PULSE_OPTIMIZATION_NONE));
    pulse_train(model, 15000, 4, (pulse_train_hyper_args_t) {2, 0.1}, PULSE_LOSS_MSE, (PULSE_DATA*)x, (PULSE_DATA*)y);

    printf("TRAIN RESULT\n");
    for (int i = 0; i < 4; i++) {
        printf("Entrada: %d %d, Output: %f\n", (int)x[i][0], (int)x[i][1], pulse_foward(model.layers, x[i])[0]);
    }

    pulse_free(&model);
}
```
## Notes:
* The project requires C standard libraries. If using non-compiled files, include them in your compilation.

* Feel free to send ideas, suggestions, questions, and requests.
