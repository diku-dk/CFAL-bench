#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_short
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=0:10:00
#SBATCH --output=quickhull_baseline-cpu.out
#SBATCH --job-name=quickhull_baseline

set -e

export OMP_PLACES="0:32:1"
export OMP_PROC_BIND=true

make clean && make

RUNS=10

numactl --interleave all ./hull ../input/100M_circle.dat $RUNS | tee quickhull_baseline_cpu32_circle.runtimes

numactl --interleave all ./hull ../input/100M_rectangle.dat $RUNS | tee quickhull_baseline_cpu32_rectangle.runtimes

numactl --interleave all ./hull ../input/100M_quadratic.dat $RUNS | tee quickhull_baseline_cpu32_quadratic.runtimes

OMP_NUM_THREADS=1 ./hull ../input/100M_circle.dat $RUNS | tee quickhull_baseline_cpu1_circle.runtimes

OMP_NUM_THREADS=1 ./hull ../input/100M_rectangle.dat $RUNS | tee quickhull_baseline_cpu1_rectangle.runtimes

OMP_NUM_THREADS=1 ./hull ../input/100M_quadratic.dat $RUNS | tee quickhull_baseline_cpu1_quadratic.runtimes
