CC = gcc
FLAGS = -Ofast -march=native -mtune=native -Wall -Wextra
DBUG_FLAGS = -fsanitize=address -fsanitize=undefined -g
FLAGS += $(DBUG_FLAGS)
LFLAGS = -lm -lblis64

.PHONY: all clean

all: seq mt alg1

seq: bin/flash
mt: bin/flash_mt
alg1: bin/flash_alg1

bin/flash: src/flash.c
	$(CC) $(FLAGS) $< -o $@ $(LFLAGS)

bin/flash_mt: src/flash_mt.c
	$(CC) $(FLAGS) $< -o $@ $(LFLAGS) -fopenmp
	
bin/flash_alg1: src/flash_alg1.c
	$(CC) $(FLAGS) $< -o $@ -lm

clean:
	$(RM) bin/*
