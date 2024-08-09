

CFLAGS = -O4 -march=native -lm -fopenmp -g -Wall

default:
	gcc ${CFLAGS} MNIST.c -o MNIST -L/home/pj/C/NeuroPulse/ -lPULSE -Wl,-rpath=../../

clone:
	curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
	curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz;
	curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz;
	curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz;
	gzip -dk *.gz
	rm *.gz

clear:
	rm train*
	rm t10k*
	rm MNIST
