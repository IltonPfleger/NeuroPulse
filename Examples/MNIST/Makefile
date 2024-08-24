
CC = gcc

CFLAGS = -O4 -march=native -lm -fopenmp -g -Wall -L$(shell pwd)/../../ -lPULSE -Wl,-rpath=../../

default:
	@if [ ! -f ./train* ]; then \
		make download; \
		fi;
	$(CC) ${CFLAGS} MNIST.c -o MNIST

download:
	curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
	curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz;
	curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz;
	curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz;
	gzip -dk *.gz
	rm *.gz

clean:
	rm train*
	rm t10k*
	rm MNIST
