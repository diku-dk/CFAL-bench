CC     = gcc
FORT   = gfortran
FLAGS  = -Ofast -march=native -mtune=native
LFLAGS = -lm -fopenmp

all: bin/nbody_c bin/nbody_f90 bin/nbody_cuda

bin/%_c: src/%.c
	$(CC) $(FLAGS) $< -o $@ $(LFLAGS)

bin/%_f90: src/%.f90
	$(FORT) $(FLAGS) $< -o $@ $(LFLAGS)

bin/%_cuda: src/%.cu
	nvcc -O3 $< -o $@
