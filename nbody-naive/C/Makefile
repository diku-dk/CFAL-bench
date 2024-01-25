CC=gcc
FORT=gfortran
CFLAGS=-Ofast -march=native -mtune=native -Wall -Wextra -pedantic
FFLAGS=-Ofast -march=native -mtune=native -Wall -Wextra -pedantic

.PHONY: all bench_seq bench_omp clean

all: nbody nbody_bench nbody_omp fortran

nbody: nbody.c
	$(CC) -o $@ $^ $(CFLAGS) -lm

fortran: nbody.f90
	$(FORT) -o $@ $^ $(FFLAGS)

nbody_bench: nbody_bench.c
	$(CC) -o $@ $^ $(CFLAGS) -lm

nbody_omp: nbody_omp.c
	$(CC) -o $@ $^ -fopenmp $(CFLAGS) -lm -fopenmp

bench_seq:
	sbatch bench_seq.sh 10000 10 10 out

bench_omp:
	sbatch bench_omp.sh 10000 10 10 out 32

clean:
	$(RM) nbody nbody_bench nbody_omp fortran
