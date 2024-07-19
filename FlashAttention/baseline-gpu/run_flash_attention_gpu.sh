#!/bin/sh

nvcc -O3 -o s2mm_cuda s2mm_cuda_valid.cu -diag-suppress=1650

./s2mm_cuda 64 16384
./s2mm_cuda 64 32768
./s2mm_cuda 128  8192
./s2mm_cuda 128 16384
