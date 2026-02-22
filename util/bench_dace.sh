#!/bin/bash

#SBATCH --partition=csmpi_fpga_long
#SBATCH --job-name=cfal-dace
#SBATCH --time=180:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G

export OMP_PLACES=cores
export OMP_PROC_BIND=true

export DACE_compiler_build_type="Release"
export DACE_compiler_allow_view_arguments=1
export DACE_compiler_cuda_default_block_size="128,1,1"
# export DACE_compiler_cuda_default_block_size="64,1,1"
export DACE_library_blas_default_implementation="MKL"


# If the script is used improperly, prints a short explanation
# and exits
if [ "$#" -lt 1 ]; then
    printf 'Usage: bench_dace.sh BENCHMARK_NAME\n\n' >&2
    printf '\tBENCHMARK_NAME: Benchmark to run\n' >&2
    exit 1
fi

bench="$1"                         # Program to benchmark

if [[ "$bench" == "FlashAttention" ]]; then
    OMP_NUM_THREADS=1 numactl --interleave all env time -v python FlashAttention/dace/bench_flash_attention_dace_cpu.py
    OMP_NUM_THREADS=32 numactl --interleave all env time -v python FlashAttention/dace/bench_flash_attention_dace_cpu.py
    python FlashAttention/dace/bench_flash_attention_dace_gpu.py
elif [[ "$bench" == "LocVolCalib" ]]; then
    OMP_NUM_THREADS=1 numactl --interleave all env time -v python LocVolCalib/dace/LocVolCalib_2.py --target cpu --size S
    OMP_NUM_THREADS=32 numactl --interleave all env time -v python LocVolCalib/dace/LocVolCalib_2.py --target cpu --size S
    python LocVolCalib/dace/LocVolCalib_2.py --target gpu --size S
    OMP_NUM_THREADS=1 numactl --interleave all env time -v python LocVolCalib/dace/LocVolCalib_2.py --target cpu --size M
    OMP_NUM_THREADS=32 numactl --interleave all env time -v python LocVolCalib/dace/LocVolCalib_2.py --target cpu --size M
    python LocVolCalib/dace/LocVolCalib_2.py --target gpu --size M
    OMP_NUM_THREADS=1 numactl --interleave all env time -v python LocVolCalib/dace/LocVolCalib_2.py --target cpu --size L
    OMP_NUM_THREADS=32 numactl --interleave all env time -v python LocVolCalib/dace/LocVolCalib_2.py --target cpu --size L
    python LocVolCalib/dace/LocVolCalib_2.py --target gpu --size L
elif [[ "$bench" == "MG" ]]; then
    OMP_NUM_THREADS=1 numactl --interleave all env time -v python MG/dace/mg.py -c A -f dace_cpu
    OMP_NUM_THREADS=32 numactl --interleave all env time -v python MG/dace/mg.py -c A -f dace_cpu
    python MG/dace/mg.py -c A -f dace_gpu
    OMP_NUM_THREADS=1 numactl --interleave all env time -v python MG/dace/mg.py -c B -f dace_cpu
    OMP_NUM_THREADS=32 numactl --interleave all env time -v python MG/dace/mg.py -c B -f dace_cpu
    python MG/dace/mg.py -c B -f dace_gpu
    OMP_NUM_THREADS=1 numactl --interleave all env time -v python MG/dace/mg.py -c C -f dace_cpu
    OMP_NUM_THREADS=32 numactl --interleave all env time -v python MG/dace/mg.py -c C -f dace_cpu
    python MG/dace/mg.py -c C -f dace_gpu
elif [[ "$bench" == "nbody-naive" ]]; then
    OMP_NUM_THREADS=1 numactl --interleave all env time -v python nbody-naive/dace/nbody_dace_cpu.py -N 1000 -iterations 100000
    OMP_NUM_THREADS=32 numactl --interleave all env time -v python nbody-naive/dace/nbody_dace_cpu.py -N 1000 -iterations 100000
    python nbody-naive/dace/nbody_dace_gpu.py -N 1000 -iterations 100000
    OMP_NUM_THREADS=1 numactl --interleave all env time -v python nbody-naive/dace/nbody_dace_cpu.py -N 10000 -iterations 1000
    OMP_NUM_THREADS=32 numactl --interleave all env time -v python nbody-naive/dace/nbody_dace_cpu.py -N 10000 -iterations 1000
    python nbody-naive/dace/nbody_dace_gpu.py -N 10000 -iterations 1000
    OMP_NUM_THREADS=1 numactl --interleave all env time -v python nbody-naive/dace/nbody_dace_cpu.py -N 100000 -iterations 10
    OMP_NUM_THREADS=32 numactl --interleave all env time -v python nbody-naive/dace/nbody_dace_cpu.py -N 100000 -iterations 10
    python nbody-naive/dace/nbody_dace_gpu.py -N 100000 -iterations 10
else
    printf 'Unknown benchmark: %s\n' "$bench" >&2
    exit 1
fi
