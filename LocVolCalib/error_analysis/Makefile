FFLAGS = -O3 -march=native -mtune=native -fbounds-check -fsanitize=address -fsanitize=undefined -Wall -Wextra -ggdb

all: tridiagonal

input:
	(cd ../baseline-FinPar/OrigCpp && make print)
	cp ../baseline-FinPar/OrigCpp/*.bin .

tridiagonal: tridiagonal.f90
	gfortran  $(FLAGS) -o $@ $< -llapack

clean:
	$(RM) tridiagonal *.bin
