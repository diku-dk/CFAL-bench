#!/bin/sh
#
# Run all Futhark benchmarks. Assumes 'futhark' is on PATH and is
# operational with the 'cuda', 'multicore', and 'ispc' backends.
#
# You can submit this script as a slurm job. On the shared system, do:
#
# $ srun -t 10:00 -p csmpi_fpga_short -c 32 --gres=gpu:nvidia_a30:1 ./benchmark_futhark.sh

make -C LocVolCalib/futhark run_ispc
make -C LocVolCalib/futhark run_cuda

make -C quickhull/futhark run_multicore
make -C quickhull/futhark run_cuda

make -C nbody-naive/futhark run_multicore
make -C nbody-naive/futhark run_cuda
