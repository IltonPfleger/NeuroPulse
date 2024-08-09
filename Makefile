

CFLAGS = -O4 -march=native -lm -fopenmp -D__PULSE_CFLAGS_CacheLineSize=`getconf LEVEL1_DCACHE_LINESIZE` -g -Wall


output: Convolutional.o Layer.o MaxPool.o Dense.o Activations.o PULSE.o Loss.o
	gcc ${CFLAGS} Convolutional.o Layer.o MaxPool.o Dense.o Activations.o Loss.o PULSE.o -shared -o libPULSE.so
	@echo

Activations.o: Activations.c
	gcc ${CFLAGS} -c Activations.c -o Activations.o

Convolutional.o: Convolutional.c
	gcc ${CFLAGS} -c Convolutional.c -o Convolutional.o

Loss.o: Loss.c
	gcc ${CFLAGS} -c Loss.c -o Loss.o

Layer.o: Layer.c
	gcc ${CFLAGS} -c Layer.c -o Layer.o

MaxPool.o: MaxPool.c
	gcc ${CFLAGS} -c MaxPool.c -o MaxPool.o

Dense.o: Dense.c
	gcc ${CFLAGS} -c Dense.c -o Dense.o

PULSE.o: PULSE.c
	gcc ${CFLAGS} -c PULSE.c -o PULSE.o

clear:
	rm *.o && rm *.so

build:
	gcc ${CFLAGS} ${MAIN}.c -o ${MAIN} -L. -lPULSE -Wl,-rpath=.
