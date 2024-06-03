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
* [ ] Custom Error Functions.

## Example
```c Xor Problem.
double x[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
double y[4][1] = {{1}, {0}, {1}, {0}};

PULSE_Layer input = PULSE_CreateLayer(2, 16, &PULSE_ReLU);
PULSE_Layer output = PULSE_CreateLayer(16, 1, &PULSE_Sigmoid);
PULSE_Connect(&input, &output);

PULSE_Train(&input, 15000, 4, (PULSE_HyperArgs){2, 0.1}, (PULSE_DataType*)x, (PULSE_DataType*)y);

printf("TRAIN RESULT\n");
for (int i = 0; i < 4; i++)
{
	input.feed(&input, x[i]);
	printf("Entrada: %d %d, Output: %.10f\n", (int)x[i][0], (int)x[i][1], output.outputs[0]);
}
```
## Notes:
* The project requires C standard libraries. If using non-compiled files, include them in your compilation.

* Feel free to send ideas, suggestions, questions, and requests.
