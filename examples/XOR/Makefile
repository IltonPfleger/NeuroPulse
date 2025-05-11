CC = gcc
CFLAGS = -O4 -Wall -Wextra -std=c23 -lm
PULSE_DIR = ../../
TARGET = XOR
SOURCES = $(TARGET).c

all: $(TARGET)

%: %.c
	$(CC) $(CFLAGS) $< -o $@ -L$(PULSE_DIR)build/ -I$(PULSE_DIR)/include -lpulse

clean:
	rm $(TARGET)

run:
	LD_LIBRARY_PATH=../../build ./$(TARGET)
