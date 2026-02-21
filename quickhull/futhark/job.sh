#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=quickhull_futhark.out
#SBATCH --job-name=quickhull_futhark

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

futhark bench --backend=multicore quickhull.fut --json quickhull_cpu32.json
../../util/futhark-json2runtimes.py quickhull_cpu32.json quickhull.fut circle \
                                    > quickhull_futhark_cpu32_A.runtimes
../../util/futhark-json2runtimes.py quickhull_cpu32.json quickhull.fut rectangle \
                                    > quickhull_futhark_cpu32_B.runtimes
../../util/futhark-json2runtimes.py quickhull_cpu32.json quickhull.fut quadratic \
                                    > quickhull_futhark_cpu32_C.runtimes

futhark bench --backend=multicore quickhull.fut --json quickhull_cpu1.json --pass-option=--num-threads=1
../../util/futhark-json2runtimes.py quickhull_cpu1.json quickhull.fut circle \
                                    > quickhull_futhark_cpu1_A.runtimes
../../util/futhark-json2runtimes.py quickhull_cpu1.json quickhull.fut rectangle \
                                    > quickhull_futhark_cpu1_B.runtimes
../../util/futhark-json2runtimes.py quickhull_cpu1.json quickhull.fut quadratic \
                                    > quickhull_futhark_cpu1_C.runtimes

futhark bench --backend=cuda quickhull.fut --json quickhull_gpu.json
../../util/futhark-json2runtimes.py quickhull_gpu.json quickhull.fut circle \
                                    > quickhull_futhark_gpu_A.runtimes
../../util/futhark-json2runtimes.py quickhull_gpu.json quickhull.fut rectangle \
                                    > quickhull_futhark_gpu_B.runtimes
../../util/futhark-json2runtimes.py quickhull_gpu.json quickhull.fut quadratic \
                                    > quickhull_futhark_gpu_C.runtimes
