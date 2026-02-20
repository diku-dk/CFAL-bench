#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=nbody_futhark.out
#SBATCH --job-name=nbody_futhark

# Ensure that the necessary tools are in PATH.
export PATH=/vol/itt/data/cfal/team-futhark/bin/:$PATH

# Fix a CUDA version.
CUDA=/usr/local/cuda
export LIBRARY_PATH=$CUDA/lib64:$CUDA/lib64/stubs
export LD_LIBRARY_PATH=$CUDA/lib64/
export CPATH=$CUDA/include

set -e

rm -f *.runtimes

futhark bench --backend=multicore nbody.fut --json nbody_cpu32.json
../../util/futhark-json2runtimes.py nbody_cpu32.json nbody.fut n=1000 \
                                    > nbody_futhark_cpu32_n1000.runtimes
../../util/futhark-json2runtimes.py nbody_cpu32.json nbody.fut n=10000 \
                                    > nbody_futhark_cpu32_n10000.runtimes
../../util/futhark-json2runtimes.py nbody_cpu32.json nbody.fut n=100000 \
                                    > nbody_futhark_cpu32_n100000.runtimes

futhark bench --backend=multicore nbody.fut --json nbody_cpu1.json --pass-option=--num-threads=1
../../util/futhark-json2runtimes.py nbody_cpu1.json nbody.fut n=1000 \
                                    > nbody_futhark_cpu1_n1000.runtimes
../../util/futhark-json2runtimes.py nbody_cpu1.json nbody.fut n=10000 \
                                    > nbody_futhark_cpu1_n10000.runtimes
../../util/futhark-json2runtimes.py nbody_cpu1.json nbody.fut n=100000 \
                                    > nbody_futhark_cpu1_n100000.runtimes

futhark bench --backend=cuda nbody.fut --json nbody_cuda.json
../../util/futhark-json2runtimes.py nbody_cuda.json nbody.fut n=1000 \
                                    > nbody_futhark_cuda_n1000.runtimes
../../util/futhark-json2runtimes.py nbody_cuda.json nbody.fut n=10000 \
                                    > nbody_futhark_cuda_n10000.runtimes
../../util/futhark-json2runtimes.py nbody_cuda.json nbody.fut n=100000 \
                                    > nbody_futhark_cuda_n100000.runtimes
