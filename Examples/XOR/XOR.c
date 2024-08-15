#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "../../Include/PULSE.h"

int main()
{
	PULSE_DataType x[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
	PULSE_DataType y[4][1] = {{1}, {0}, {1}, {0}};

	PULSE_Layer * model = PULSE_CreateModel(2,
			PULSE_DENSE, (PULSE_ARGS_DENSE){2, 128, PULSE_ACTIVATION_RELU, PULSE_OPTIMIZATION_NONE},
			PULSE_DENSE,(PULSE_ARGS_DENSE){128, 1, PULSE_ACTIVATION_RELU, PULSE_OPTIMIZATION_NONE});

	double t1 = omp_get_wtime();
	PULSE_Train(model, 15000, 4, (PULSE_HyperArgs){2, 0.1}, PULSE_LOSS_MSE, (PULSE_DataType*)x, (PULSE_DataType*)y);
	double t2 = omp_get_wtime();
	printf("%f\n", t2 - t1);

	printf("TRAIN RESULT\n");
	for (int i = 0; i < 4; i++)
	{
		printf("Entrada: %d %d, Output: %f\n", (int)x[i][0], (int)x[i][1], PULSE_Foward(model, x[i])[0]);
	}
}
