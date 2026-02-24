CUDA_CC?=80
CFLAGS=-O3 -gencode arch=compute_$(CUDA_CC),code=sm_$(CUDA_CC)

nbody: nbody.cu util.cu.h
	nvcc -o nbody nbody.cu $(CFLAGS)

run: nbody
	rm -f nbody-gpu.csv
	./nbody 1000 100000 10 >> nbody-gpu.csv
	./nbody 10000  1000 10 >> nbody-gpu.csv
	./nbody 100000   10 10 >> nbody-gpu.csv

clean:
	rm -f nbody
