#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=MG_baseline-cpu.out
#SBATCH --job-name=MG_baseline

set -e

export OMP_PLACES="0:32:1"
export OMP_PROC_BIND=true

rm -f *.runtimes

make clean
make mg CLASS=A
make mg CLASS=B
make mg CLASS=C

RUNS=15
USE_RUNS=10

for x in $(seq $RUNS); do
    bin/mg.A.x | awk '/Time in seconds/ {print $5}'
done | tail -n $USE_RUNS | tee MG_baseline_cpu32_A.runtimes

for x in $(seq $RUNS); do
    bin/mg.B.x | awk '/Time in seconds/ {print $5}'
done | tail -n $USE_RUNS | tee MG_baseline_cpu32_B.runtimes

for x in $(seq $RUNS); do
    bin/mg.C.x | awk '/Time in seconds/ {print $5}'
done | tail -n $USE_RUNS | tee MG_baseline_cpu32_C.runtimes

for x in $(seq $RUNS); do
    OMP_NUM_THREADS=1 bin/mg.A.x | awk '/Time in seconds/ {print $5}'
done | tail -n $USE_RUNS | tee MG_baseline_cpu1_A.runtimes

for x in $(seq $RUNS); do
    OMP_NUM_THREADS=1 bin/mg.B.x | awk '/Time in seconds/ {print $5}'
done | tail -n $USE_RUNS | tee MG_baseline_cpu1_B.runtimes

for x in $(seq $RUNS); do
    OMP_NUM_THREADS=1 bin/mg.C.x | awk '/Time in seconds/ {print $5}'
done | tail -n $USE_RUNS | tee MG_baseline_cpu1_C.runtimes
