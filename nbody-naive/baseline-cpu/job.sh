#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=nbody_baseline-cpu.out
#SBATCH --job-name=nbody_baseline-cpu

set -e

export OMP_PLACES="0:32:1"
export OMP_PROC_BIND=true

make clean && make

export CFLAGS='-Ofast -march=native -mtune=native'

RUNS=10

./nbody 1000 100000 $RUNS | tee nbody_baseline_cpu32_n1000.runtimes
./nbody 10000  1000 $RUNS | tee nbody_baseline_cpu32_n10000.runtimes
./nbody 100000   10 $RUNS | tee nbody_baseline_cpu32_n100000.runtimes
OMP_NUM_THREADS=1 ./nbody 1000 100000 $RUNS | tee nbody_baseline_cpu1_n1000.runtimes
OMP_NUM_THREADS=1 ./nbody 10000  1000 $RUNS | tee nbody_baseline_cpu1_n10000.runtimes
OMP_NUM_THREADS=1 ./nbody 100000   10 $RUNS | tee nbody_baseline_cpu1_n100000.runtimes
