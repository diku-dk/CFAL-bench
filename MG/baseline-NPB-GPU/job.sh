#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=mg_baseline-gpu.out
#SBATCH --job-name=mg_baseline-gpu

set -e

rm -f *.runtimes

make clean
make mg CLASS=A
make mg CLASS=B
make mg CLASS=C

RUNS=15
USE_RUNS=10

for x in $(seq $RUNS); do
    bin/mg.A | awk '/Time in seconds/ {print $5}'
done | tail -n $USE_RUNS | tee mg_baseline_gpu_A.runtimes

for x in $(seq $RUNS); do
    bin/mg.B | awk '/Time in seconds/ {print $5}'
done | tail -n $USE_RUNS | tee mg_baseline_gpu_B.runtimes

for x in $(seq $RUNS); do
    bin/mg.C | awk '/Time in seconds/ {print $5}'
done | tail -n $USE_RUNS | tee mg_baseline_gpu_C.runtimes
