C = gfortran
FLAGS = -O3 -march=native -mtune=native -Wall -ffast-math

all: fft

fft: fft.f90
	$(C) $(FLAGS) $< -o $@

run:
	./fft 256

clean:
	$(RM) fft fft_mod.mod
