output: Convolutional.o Layer.o MaxPoll.o Dense.o Activations.o PULSE.o
	gcc -O3 Convolutional.o Layer.o MaxPoll.o Dense.o Activations.o PULSE.o -shared -o PULSE.so -lm -fopenmp
	@echo

Activations.o: Activations.c
	gcc -O3 -c Activations.c -o Activations.o

Convolutional.o: Convolutional.c
	gcc -O3 -c Convolutional.c -o Convolutional.o -fopenmp

Layer.o: Layer.c
	gcc -O3 -c Layer.c -o Layer.o

MaxPoll.o: MaxPoll.c
	gcc -O3 -c MaxPoll.c -o MaxPoll.o -fopenmp

Dense.o: Dense.c
	gcc -O3 -c Dense.c -o Dense.o

PULSE.o: PULSE.c
	gcc -O3 -c PULSE.c -o PULSE.o

clear:
	rm *.o
