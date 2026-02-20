#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=mg_futhark.out

# Ensure that the necessary tools are in PATH.
export PATH=/vol/itt/data/cfal/team-futhark/bin/:$PATH

# Fix a CUDA version.
CUDA=/usr/local/cuda
export LIBRARY_PATH=$CUDA/lib64:$CUDA/lib64/stubs
export LD_LIBRARY_PATH=$CUDA/lib64/
export CPATH=$CUDA/include

set -e

rm -f *.runtimes

futhark bench --backend=multicore mg.fut --json mg_cpu32.json
../../util/futhark-json2runtimes.py mg_cpu32.json mg.fut:mgNAS "Class A" \
                                    > mg_futhark_cpu32_A.runtimes
../../util/futhark-json2runtimes.py mg_cpu32.json mg.fut:mgNAS "Class B" \
                                    > mg_futhark_cpu32_B.runtimes
../../util/futhark-json2runtimes.py mg_cpu32.json mg.fut:mgNAS "Class C" \
                                    > mg_futhark_cpu32_C.runtimes

futhark bench --backend=multicore mg.fut --json mg_cpu1.json --pass-option=--num-threads=1
../../util/futhark-json2runtimes.py mg_cpu1.json mg.fut:mgNAS "Class A" \
                                    > mg_futhark_cpu1_A.runtimes
../../util/futhark-json2runtimes.py mg_cpu1.json mg.fut:mgNAS "Class B" \
                                    > mg_futhark_cpu1_B.runtimes
../../util/futhark-json2runtimes.py mg_cpu1.json mg.fut:mgNAS "Class C" \
                                    > mg_futhark_cpu1_C.runtimes

futhark bench --backend=cuda mg.fut --json mg_cuda.json
../../util/futhark-json2runtimes.py mg_cuda.json mg.fut:mgNAS "Class A" \
                                    > mg_futhark_cuda_A.runtimes
../../util/futhark-json2runtimes.py mg_cuda.json mg.fut:mgNAS "Class B" \
                                    > mg_futhark_cuda_B.runtimes
../../util/futhark-json2runtimes.py mg_cuda.json mg.fut:mgNAS "Class C" \
                                    > mg_futhark_cuda_C.runtimes
