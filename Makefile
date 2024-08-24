CC = gcc
CFLAGS = -O4 -march=native -lm -D__PULSE_CFLAGS_CacheLineSize=`getconf LEVEL1_DCACHE_LINESIZE` -Wall -g
OBJECTS = Dense.o Activations.o PULSE.o Loss.o
LIBRARY = libPULSE.so

# Add GPU support if available
ifeq ($(shell clinfo | awk '/Device Type/ {print $$3; exit}'), GPU)
	CFLAGS += -D__PULSE_GPU_SUPPORTED -lOpenCL
endif

$(LIBRARY): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -shared -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.so

run:
	(cd Examples/$(EXAMPLE) && make)
	(cd Examples/$(EXAMPLE) && ./$(EXAMPLE))

all: $(LIBRARY)
