CC=gcc
CFLAGS=-Ofast -march=native -mtune=native -Wall -Wextra -pedantic

all: nbody nbody_bench

nbody: nbody.c
	$(CC) -o $@ $^ $(CFLAGS) -lm

nbody_bench: nbody_bench.c
	$(CC) -o $@ $^ $(CFLAGS) -lm

clean:
	$(RM) nbody nbody_bench
