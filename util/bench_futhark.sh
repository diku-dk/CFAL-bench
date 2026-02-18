#!/bin/sh
#SBATCH --partition=csmpi_fpga_long
#SBATCH --job-name=cfal-futhark
#SBATCH --time=180:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --output=bench_futhark.out
#
# Run all Futhark benchmarks. If run on the shared system at Radboud,
# automatically sets up the PATH appropriately.
#
# You should submit this script as a slurm job. On the shared system,
# do:
#
# $ sbatch util/bench_futhark.sh
#
# Note that it is important that you run this from the top level of
# the repository, as otherwise the paths will be wrong.

set -e

# Ensure that the necessary tools are in PATH.
export PATH=/vol/itt/data/cfal/team-futhark/bin/:$PATH

# Fix a CUDA version.
CUDA=/usr/local/cuda
export LIBRARY_PATH=$CUDA/lib64:$CUDA/lib64/stubs
export LD_LIBRARY_PATH=$CUDA/lib64/
export CPATH=$CUDA/include

time2flops="util/futhark-time2flops.py"

do_FlashAttention() {
    make -C FlashAttention/futhark run_multicore
    make -C FlashAttention/futhark run_cuda

    echo "FlashAttention GPU GFLOP/s"
    $time2flops FlashAttention/futhark/custom-alg1-opt_cuda.json custom-alg1-opt.fut:validate \
		'Class 16384-64 ' 68.72 \
		'Class 32768-64 ' 274.88 \
		'Class 8192-128 ' 34.36 \
		'Class 16384-128' 127.44

    echo "FlashAttention CPU GFLOP/s (1 thread)"
    $time2flops FlashAttention/futhark/custom-alg1-opt_multicore_1.json custom-alg1-opt.fut:validate \
		'Class 16384-64 ' 68.72 \
		'Class 32768-64 ' 274.88 \
		'Class 8192-128 ' 34.36 \
		'Class 16384-128' 127.44

    echo "FlashAttention CPU GFLOP/s (max threads)"
    $time2flops FlashAttention/futhark/custom-alg1-opt_multicore.json custom-alg1-opt.fut:validate \
		'Class 16384-64 ' 68.72 \
		'Class 32768-64 ' 274.88 \
		'Class 8192-128 ' 34.36 \
		'Class 16384-128' 127.44
}

do_MG() {
#    make -C MG/futhark run_multicore
    make -C MG/futhark run_cuda

    echo "MG CPU GFLOP/s (1 thread)"
    $time2flops MG/futhark/mg_multicore_1.json mg.fut:mgNAS 'Class A' 3.625 'Class B' 18.125  'Class C' 145.0

    echo "MG CPU GFLOP/s (max threads)"
    $time2flops MG/futhark/mg_multicore.json mg.fut:mgNAS 'Class A' 3.625 'Class B' 18.125  'Class C' 145.0

    echo "MG GPU GFLOP/s"
    $time2flops MG/futhark/mg_cuda.json mg.fut:mgNAS 'Class A' 3.625  'Class B' 18.125 'Class C' 145.0
}

do_LocVolCalib() {
    make -C LocVolCalib/futhark run_ispc
    make -C LocVolCalib/futhark run_cuda
}

do_quickhull() {
    make -C quickhull/futhark run_multicore
    make -C quickhull/futhark run_cuda
}

do_nbody() {
    make -C nbody-naive/futhark run_multicore
    make -C nbody-naive/futhark run_cuda

    echo "N-body CPU GFLOP/s (1 thread)"
    $time2flops nbody-naive/futhark/nbody_multicore_1.json nbody.fut 'n=1000' 1800 'n=10000' 1800 'n=100000' 1800

    echo "N-body CPU GFLOP/s (max threads)"
    $time2flops nbody-naive/futhark/nbody_multicore.json nbody.fut 'n=1000' 1800 'n=10000' 1800 'n=100000' 1800

    echo "N-body GPU GFLOP/s"
    $time2flops nbody-naive/futhark/nbody_cuda.json nbody.fut 'n=1000' 1800 'n=10000' 1800 'n=100000' 1800
}

do_FlashAttention
do_MG
do_LocVolCalib
do_quickhull
do_nbody
