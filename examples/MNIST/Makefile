CC = gcc
CFLAGS = -O4 -fopenmp -Wall -std=c23 -lm
PULSE_DIR = $(shell dirname $(shell dirname $(shell pwd)))
SOURCES = $(shell find . -type f -name "*.c")
TARGET = $(SOURCES:.c=)

all: $(TARGET)

%: %.c
	@if [ ! -f ./train* ]; then \
		make download; \
		fi;
	$(CC) $(CFLAGS) $< -o $@ -Wl,-rpath=$(PULSE_DIR) -L$(PULSE_DIR) -I$(PULSE_DIR)/include -lpulse


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
	rm $(TARGET)
