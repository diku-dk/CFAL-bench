CC = gcc
FLAGS = -O3 -march=native -mtune=native -Wall
LFLAGS = -lm

SRCS = $(wildcard src/*.c)
BINS = $(patsubst src/%.c, bin/%, $(SRCS))

all: $(BINS)

bin/%: src/%.c
	$(CC) $(FLAGS) $^ -o $@ $(LFLAGS)

clean:
	$(RM) bin/*
