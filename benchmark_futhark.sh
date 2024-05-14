#!/bin/sh
#
# Run all Futhark benchmarks. Assumes 'futhark' is on PATH and is
# operational with the 'cuda', 'multicore', and 'ispc' backends.
#
# You can submit this script as a slurm job.

make -C LocVolCalib/futhark run_ispc
make -C LocVolCalib/futhark run_cuda

make -C quickhull/futhark run_multicore
make -C quickhull/futhark run_cuda

make -C nbody-naive/futhark run_multicore
make -C nbody-naive/futhark run_cuda
