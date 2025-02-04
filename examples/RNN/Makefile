CC = gcc
CFLAGS = -O4 -fopenmp -Wall
PULSE_DIR = $(shell dirname $(shell dirname $(shell pwd)))
SOURCES = $(shell find . -type f -name "*.c")
TARGET = $(SOURCES:.c=)

all: $(TARGET)

%: %.c
	$(CC) $(CFLAGS) $< -o $@ -Wl,-rpath=$(PULSE_DIR) -L$(PULSE_DIR) -I$(PULSE_DIR)/include -lpulse

clean:
	rm -f $(TARGET)
