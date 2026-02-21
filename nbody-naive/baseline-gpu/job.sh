#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=nbody_baseline-gpu.out
#SBATCH --job-name=nbody_baseline-gpu

set -e

make clean && make
rm -f *.runtimes

RUNS=10

./nbody 1000 100000 $RUNS | tee nbody_baseline_cuda_n1000.runtimes
./nbody 10000  1000 $RUNS | tee nbody_baseline_cuda_n10000.runtimes
./nbody 100000   10 $RUNS | tee nbody_baseline_cuda_n100000.runtimes
