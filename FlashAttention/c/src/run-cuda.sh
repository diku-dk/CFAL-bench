#!/bin/sh

nvcc -O3 -o flash-cuda flash_alg1_cuda.cu -diag-suppress=1650

./flash-cuda 128  8192 8192
./flash-cuda 128 16384 8192
./flash-cuda  64  8192 4096
./flash-cuda  64 16384 4096
