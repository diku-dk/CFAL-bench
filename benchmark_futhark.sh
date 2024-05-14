#!/bin/sh
#
# Run all Futhark benchmarks. Assumes 'futhark' is on PATH and is
# operational with the 'cuda', 'multicore', and 'ispc' backends.
#
# You can submit this script as a slurm job. On the shared system, do:
#
# $ srun -t 10:00 -p csmpi_fpga_short -c 32 --gres=gpu:nvidia_a30:1 ./benchmark_futhark.sh

make -C MG/futhark run_multicore
make -C MG/futhark run_cuda

echo MG CPU GFLOP/s
./futhark-time2flops.py mg/futhark/mg_multicore.json mg.fut 'class A' 3.625
./futhark-time2flops.py mg/futhark/mg_multicore.json mg.fut 'class B' 18.125
./futhark-time2flops.py mg/futhark/mg_multicore.json mg.fut 'class C' 145.0

echo MG GPU GFLOP/s
echo MG CPU GFLOP/s
./futhark-time2flops.py mg/futhark/mg_cuda.json mg.fut 'class A' 3.625
./futhark-time2flops.py mg/futhark/mg_cuda.json mg.fut 'class B' 18.125
./futhark-time2flops.py mg/futhark/mg_cuda.json mg.fut 'class C' 145.0

make -C LocVolCalib/futhark run_ispc
make -C LocVolCalib/futhark run_cuda

make -C quickhull/futhark run_multicore
make -C quickhull/futhark run_cuda

make -C nbody-naive/futhark run_multicore
make -C nbody-naive/futhark run_cuda

echo N-body CPU GFLOP/s
./futhark-time2flops.py nbody-naive/futhark/nbody_multicore.json nbody.fut 'k=10, n=1000' 0.18012
./futhark-time2flops.py nbody-naive/futhark/nbody_multicore.json nbody.fut 'k=10, n=10000' 18.0012
./futhark-time2flops.py nbody-naive/futhark/nbody_multicore.json nbody.fut 'k=10, n=100000' 1800.0012

echo N-body GPU GFLOP/s
./futhark-time2flops.py nbody-naive/futhark/nbody_cuda.json nbody.fut 'k=10, n=1000' 0.18012
./futhark-time2flops.py nbody-naive/futhark/nbody_cuda.json nbody.fut 'k=10, n=10000' 18.0012
./futhark-time2flops.py nbody-naive/futhark/nbody_cuda.json nbody.fut 'k=10, n=100000' 1800.0012
