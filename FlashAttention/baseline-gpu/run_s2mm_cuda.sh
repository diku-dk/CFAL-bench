#!/bin/sh

nvcc -O3 -o s2mm_cuda s2mm_cuda.cu -diag-suppress=1650

./s2mm_cuda 128  8192
./s2mm_cuda 128 16384
