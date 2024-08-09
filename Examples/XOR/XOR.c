#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "../../Include/PULSE.h"

int main()
{
	PULSE_DataType x[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
	PULSE_DataType y[4][1] = {{1}, {0}, {1}, {0}};

	PULSE_Layer input = PULSE_CreateDenseLayer(2, 256, &PULSE_ReLU, PULSE_OPTIMIZATION_SIMD);
	PULSE_Layer output = PULSE_CreateDenseLayer(256, 1, &PULSE_Sigmoid, PULSE_OPTIMIZATION_SIMD);
	PULSE_Connect(&input, &output);

	double t1 = omp_get_wtime();
	PULSE_Train(&input, 15000, 4, (PULSE_HyperArgs){2, 0.1}, (PULSE_DataType*)x, (PULSE_DataType*)y);
	double t2 = omp_get_wtime();
	printf("%f\n", t2 - t1);

	printf("TRAIN RESULT\n");
	for (int i = 0; i < 4; i++)
	{
		PULSE_Foward(&input, x[i]);
		printf("Entrada: %d %d, Output: %f\n", (int)x[i][0], (int)x[i][1], output.outputs[0]);
	}
}
