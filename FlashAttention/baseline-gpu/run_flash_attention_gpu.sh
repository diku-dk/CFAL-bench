#!/bin/sh

nvcc -O3 -o flash_attention_gpu flash_attention_gpu.cu -diag-suppress=1650

./flash_attention_gpu 64 16384
./flash_attention_gpu 64 32768
./flash_attention_gpu 128  8192
./flash_attention_gpu 128 16384
