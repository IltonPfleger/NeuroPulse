CC = gcc
CFLAGS = -O4 -march=native -I ./include -lm -D__PULSE_CFLAGS_CacheLineSize=`getconf LEVEL1_DCACHE_LINESIZE` -Wall -g -fopenmp
OBJECTS = dense.o activations.o pulse.o loss.o
LIBRARY = libpulse.so

$(LIBRARY): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -shared -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o *.so

run:
	(cd examples/$(EXAMPLE) && make)
	(cd examples/$(EXAMPLE) && ./$(EXAMPLE))

all: $(LIBRARY)
