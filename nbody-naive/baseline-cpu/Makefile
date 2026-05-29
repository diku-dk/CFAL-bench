CC=cc
CFLAGS=-Wall -fopenmp -lm -Ofast -march=native -mtune=native

nbody: nbody.c
	$(CC) -o nbody nbody.c $(CFLAGS)

clean:
	rm -rf nbody *.runtimes
