CC=cc
CFLAGS=-O3 -Wall -fopenmp -lm -ffast-math

nbody: nbody.c
	$(CC) -o nbody nbody.c $(CFLAGS)

run: nbody
	rm -f nbody-cpu.csv
	./nbody 1000 100000 10 >> nbody-cpu.csv
	./nbody 10000  1000 10 >> nbody-cpu.csv
	./nbody 100000   10 10 >> nbody-cpu.csv
