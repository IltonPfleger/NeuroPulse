CC = gcc
CFLAGS = -O4 -march=native -I ./include -lm -Wall -Wextra -fPIC
#CFLAGS += -D__PULSE_CFLAGS_CacheLineSize=`getconf LEVEL1_DCACHE_LINESIZE`

SOURCES = $(shell find . -path "./examples/*" -prune -o -type f -name "*.c" -print)

OBJECTS = $(SOURCES:.c=.o)
OBJECTS := $(subst ./,build/,$(OBJECTS))

LIBRARY = libpulse.so

all: $(LIBRARY)

$(LIBRARY): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -shared -o $@

build/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf build
	find . -type f -name "*.so" -type f -name "*.o" -exec rm {} +

run:
	(cd examples/$(EXAMPLE) && make)
	(cd examples/$(EXAMPLE) && ./$(EXAMPLE))

