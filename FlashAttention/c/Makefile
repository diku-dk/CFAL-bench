CC = gcc
FLAGS = -Ofast -march=native -mtune=native -Wall -Wextra
DBUG_FLAGS = -fsanitize=address -fsanitize=undefined -g
FLAGS += $(DBUG_FLAGS)
LFLAGS = -lm -lblis64

.PHONY: all clean

all: seq mt

seq: bin/flash
mt: bin/flash_mt

bin/flash: src/flash.c
	$(CC) $(FLAGS) $< -o $@ $(LFLAGS)

bin/flash_mt: src/flash_mt.c
	$(CC) $(FLAGS) $< -o $@ $(LFLAGS) -fopenmp

clean:
	$(RM) bin/*
