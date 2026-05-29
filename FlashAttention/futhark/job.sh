#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=flashattention_futhark.out
#SBATCH --job-name=flashattention_futhark

# Ensure that the necessary tools are in PATH.
export PATH=/vol/itt/data/cfal/team-futhark/bin/:$PATH

# Fix a CUDA version.
CUDA=/usr/local/cuda
export LIBRARY_PATH=$CUDA/lib64:$CUDA/lib64/stubs
export LD_LIBRARY_PATH=$CUDA/lib64/
export CPATH=$CUDA/include

export CFLAGS='-Ofast -march=native -mtune=native'

set -e

rm -f *.runtimes

numactl --interleave all futhark bench --backend=multicore custom-alg1-opt.fut -e validate --json FlashAttention_cpu32.json --no-tuning
../../util/futhark-json2runtimes.py FlashAttention_cpu32.json custom-alg1-opt.fut:validate 'Class 16384-64 ' \
                                    > FlashAttention_futhark_cpu32_d64-N16384.runtimes
../../util/futhark-json2runtimes.py FlashAttention_cpu32.json custom-alg1-opt.fut:validate 'Class 32768-64 ' \
                                    > FlashAttention_futhark_cpu32_d64-N32768.runtimes
../../util/futhark-json2runtimes.py FlashAttention_cpu32.json custom-alg1-opt.fut:validate 'Class 8192-128 ' \
                                    > FlashAttention_futhark_cpu32_d128-N8192.runtimes
../../util/futhark-json2runtimes.py FlashAttention_cpu32.json custom-alg1-opt.fut:validate 'Class 16384-128' \
                                    > FlashAttention_futhark_cpu32_d128-N16384.runtimes
../../util/futhark-json2mem.py FlashAttention_cpu32.json custom-alg1-opt.fut:validate 'Class 16384-64 ' \
                                    > FlashAttention_futhark_cpu32_d64-N16384.bytes
../../util/futhark-json2mem.py FlashAttention_cpu32.json custom-alg1-opt.fut:validate 'Class 32768-64 ' \
                                    > FlashAttention_futhark_cpu32_d64-N32768.bytes
../../util/futhark-json2mem.py FlashAttention_cpu32.json custom-alg1-opt.fut:validate 'Class 8192-128 ' \
                                    > FlashAttention_futhark_cpu32_d128-N8192.bytes
../../util/futhark-json2mem.py FlashAttention_cpu32.json custom-alg1-opt.fut:validate 'Class 16384-128' \
                                    > FlashAttention_futhark_cpu32_d128-N16384.bytes

futhark bench --backend=multicore custom-alg1-opt.fut -e validate --json FlashAttention_cpu1.json --no-tuning --pass-option=--num-threads=1
../../util/futhark-json2runtimes.py FlashAttention_cpu1.json custom-alg1-opt.fut:validate 'Class 16384-64 ' \
                                    > FlashAttention_futhark_cpu1_d64-N16384.runtimes
../../util/futhark-json2runtimes.py FlashAttention_cpu1.json custom-alg1-opt.fut:validate 'Class 32768-64 ' \
                                    > FlashAttention_futhark_cpu1_d64-N32768.runtimes
../../util/futhark-json2runtimes.py FlashAttention_cpu1.json custom-alg1-opt.fut:validate 'Class 8192-128 ' \
                                    > FlashAttention_futhark_cpu1_d128-N8192.runtimes
../../util/futhark-json2runtimes.py FlashAttention_cpu1.json custom-alg1-opt.fut:validate 'Class 16384-128' \
                                    > FlashAttention_futhark_cpu1_d128-N16384.runtimes
../../util/futhark-json2mem.py FlashAttention_cpu1.json custom-alg1-opt.fut:validate 'Class 16384-64 ' \
                                    > FlashAttention_futhark_cpu1_d64-N16384.bytes
../../util/futhark-json2mem.py FlashAttention_cpu1.json custom-alg1-opt.fut:validate 'Class 32768-64 ' \
                                    > FlashAttention_futhark_cpu1_d64-N32768.bytes
../../util/futhark-json2mem.py FlashAttention_cpu1.json custom-alg1-opt.fut:validate 'Class 8192-128 ' \
                                    > FlashAttention_futhark_cpu1_d128-N8192.bytes
../../util/futhark-json2mem.py FlashAttention_cpu1.json custom-alg1-opt.fut:validate 'Class 16384-128' \
                                    > FlashAttention_futhark_cpu1_d128-N16384.bytes

futhark bench --backend=cuda custom-alg1-opt.fut -e validate --json FlashAttention_gpu.json
../../util/futhark-json2runtimes.py FlashAttention_gpu.json custom-alg1-opt.fut:validate 'Class 16384-64 ' \
                                    > FlashAttention_futhark_gpu_d64-N16384.runtimes
../../util/futhark-json2runtimes.py FlashAttention_gpu.json custom-alg1-opt.fut:validate 'Class 32768-64 ' \
                                    > FlashAttention_futhark_gpu_d64-N32768.runtimes
../../util/futhark-json2runtimes.py FlashAttention_gpu.json custom-alg1-opt.fut:validate 'Class 8192-128 ' \
                                    > FlashAttention_futhark_gpu_d128-N8192.runtimes
../../util/futhark-json2runtimes.py FlashAttention_gpu.json custom-alg1-opt.fut:validate 'Class 16384-128' \
                                    > FlashAttention_futhark_gpu_d128-N16384.runtimes
../../util/futhark-json2mem.py FlashAttention_gpu.json custom-alg1-opt.fut:validate 'Class 16384-64 ' \
                                    > FlashAttention_futhark_gpu_d64-N16384.bytes
../../util/futhark-json2mem.py FlashAttention_gpu.json custom-alg1-opt.fut:validate 'Class 32768-64 ' \
                                    > FlashAttention_futhark_gpu_d64-N32768.bytes
../../util/futhark-json2mem.py FlashAttention_gpu.json custom-alg1-opt.fut:validate 'Class 8192-128 ' \
                                    > FlashAttention_futhark_gpu_d128-N8192.bytes
../../util/futhark-json2mem.py FlashAttention_gpu.json custom-alg1-opt.fut:validate 'Class 16384-128' \
                                    > FlashAttention_futhark_gpu_d128-N16384.bytes
