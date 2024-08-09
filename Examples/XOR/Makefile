

CFLAGS = -O4 -march=native -lm -fopenmp -g

default:
	gcc ${CFLAGS} XOR.c -o XOR -L/home/pj/C/NeuroPulse/ -lPULSE -Wl,-rpath=../../

clear:
	rm XOR
