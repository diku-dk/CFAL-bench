#!/bin/sh
#SBATCH --partition=csmpi_fpga_short
#SBATCH --job-name=cfal-futhark
#SBATCH --time=10:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#
# Run all Futhark benchmarks. Assumes 'futhark' is on PATH and is
# operational with the 'cuda', 'multicore', and 'ispc' backends.
#
# You can submit this script as a slurm job. On the shared system, do:
#
# $ sbatch ./benchmark_futhark.sh

set -e

make -C MG/futhark run_multicore
make -C MG/futhark run_cuda

echo MG CPU GFLOP/s
./futhark-time2flops.py MG/futhark/mg_multicore.json mg.fut:mgNAS 'Class A' 3.625 'Class B' 18.125  'Class C' 145.0

echo MG GPU GFLOP/s
./futhark-time2flops.py MG/futhark/mg_cuda.json mg.fut:mgNAS 'Class A' 3.625  'Class B' 18.125 'Class C' 145.0

make -C LocVolCalib/futhark run_ispc
make -C LocVolCalib/futhark run_cuda

make -C quickhull/futhark run_multicore
make -C quickhull/futhark run_cuda

make -C nbody-naive/futhark run_multicore
make -C nbody-naive/futhark run_cuda

echo N-body CPU GFLOP/s
./futhark-time2flops.py nbody-naive/futhark/nbody_multicore.json nbody.fut 'k=10, n=1000' 0.18012 'k=10, n=10000' 18.0012 'k=10, n=100000' 1800.0012

echo N-body GPU GFLOP/s
./futhark-time2flops.py nbody-naive/futhark/nbody_cuda.json nbody.fut 'k=10, n=1000' 0.18012 'k=10, n=10000' 18.0012 'k=10, n=100000' 1800.0012
