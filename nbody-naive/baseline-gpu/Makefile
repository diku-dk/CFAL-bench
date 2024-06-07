CUDA_CC?=80
CFLAGS=-O3 -gencode arch=compute_$(CUDA_CC),code=sm_$(CUDA_CC)

nbody: nbody.cu
	nvcc -o nbody nbody.cu $(CFLAGS)
