#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=nbody_futhark.out
#SBATCH --job-name=nbody_futhark

# Ensure that the necessary tools are in PATH.
export PATH=/vol/itt/data/cfal/team-futhark/bin/:$PATH

# Fix a CUDA version.
CUDA=/usr/local/cuda
export LIBRARY_PATH=$CUDA/lib64:$CUDA/lib64/stubs
export LD_LIBRARY_PATH=$CUDA/lib64/
export CPATH=$CUDA/include

export CFLAGS='-Ofast -march=native -mtune=native'

set -e

rm -f *.runtimes

numactl --interleave all futhark bench --backend=multicore nbody.fut --json nbody_cpu32.json
../../util/futhark-json2runtimes.py nbody_cpu32.json nbody.fut n=1000 \
                                    > nbody_futhark_cpu32_n1000.runtimes
../../util/futhark-json2runtimes.py nbody_cpu32.json nbody.fut n=10000 \
                                    > nbody_futhark_cpu32_n10000.runtimes
../../util/futhark-json2runtimes.py nbody_cpu32.json nbody.fut n=100000 \
                                    > nbody_futhark_cpu32_n100000.runtimes
../../util/futhark-json2mem.py nbody_cpu32.json nbody.fut n=1000 \
                                    > nbody_futhark_cpu32_n1000.bytes
../../util/futhark-json2mem.py nbody_cpu32.json nbody.fut n=10000 \
                                    > nbody_futhark_cpu32_n10000.bytes
../../util/futhark-json2mem.py nbody_cpu32.json nbody.fut n=100000 \
                                    > nbody_futhark_cpu32_n100000.bytes

futhark bench --backend=multicore nbody.fut --json nbody_cpu1.json --pass-option=--num-threads=1
../../util/futhark-json2runtimes.py nbody_cpu1.json nbody.fut n=1000 \
                                    > nbody_futhark_cpu1_n1000.runtimes
../../util/futhark-json2runtimes.py nbody_cpu1.json nbody.fut n=10000 \
                                    > nbody_futhark_cpu1_n10000.runtimes
../../util/futhark-json2runtimes.py nbody_cpu1.json nbody.fut n=100000 \
                                    > nbody_futhark_cpu1_n100000.runtimes
../../util/futhark-json2mem.py nbody_cpu1.json nbody.fut n=1000 \
                                    > nbody_futhark_cpu1_n1000.bytes
../../util/futhark-json2mem.py nbody_cpu1.json nbody.fut n=10000 \
                                    > nbody_futhark_cpu1_n10000.bytes
../../util/futhark-json2mem.py nbody_cpu1.json nbody.fut n=100000 \
                                    > nbody_futhark_cpu1_n100000.bytes

futhark bench --backend=cuda nbody.fut --json nbody_gpu.json
../../util/futhark-json2runtimes.py nbody_gpu.json nbody.fut n=1000 \
                                    > nbody_futhark_gpu_n1000.runtimes
../../util/futhark-json2runtimes.py nbody_gpu.json nbody.fut n=10000 \
                                    > nbody_futhark_gpu_n10000.runtimes
../../util/futhark-json2runtimes.py nbody_gpu.json nbody.fut n=100000 \
                                    > nbody_futhark_gpu_n100000.runtimes
../../util/futhark-json2mem.py nbody_gpu.json nbody.fut n=1000 \
                                    > nbody_futhark_gpu_n1000.bytes
../../util/futhark-json2mem.py nbody_gpu.json nbody.fut n=10000 \
                                    > nbody_futhark_gpu_n10000.bytes
../../util/futhark-json2mem.py nbody_gpu.json nbody.fut n=100000 \
                                    > nbody_futhark_gpu_n100000.bytes
