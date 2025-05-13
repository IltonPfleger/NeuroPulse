CC = gcc
CFLAGS = -Wall -Wextra -Werror -Wpedantic
CFLAGS = -O4 -march=native -Iinclude -lm -fPIC

SOURCES = $(shell find . -type f -name '*.c' | grep -v examples)
OBJECTS = $(shell find . -type f -name '*.c' | grep -v examples | sed 's/.c$$/.o/g' | sed 's/^/build\//g')
LIBRARY = build/libpulse.so

all: $(LIBRARY)

$(LIBRARY): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -shared -o $(LIBRARY)

build/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf build/*

run:
	(cd examples/$(EXAMPLE) && make)
	(cd examples/$(EXAMPLE) && ./$(EXAMPLE))

