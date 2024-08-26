CC := gcc
CFLAGS := -O4 -fopenmp -Wall
PULSE_DIR := $(shell dirname $(shell dirname $(shell pwd)))

default:
	$(CC) ${CFLAGS} XOR.c -o XOR -Wl,-rpath=$(PULSE_DIR) -L$(PULSE_DIR) -lPULSE

clean:
	rm XOR
