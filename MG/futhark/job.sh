#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=MG_futhark.out
#SBATCH --job-name=MG_futhark

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

numactl --interleave all futhark bench --backend=multicore mg.fut --json mg_cpu32.json
../../util/futhark-json2runtimes.py mg_cpu32.json mg.fut:mgNAS "Class A" \
                                    > MG_futhark_cpu32_A.runtimes
../../util/futhark-json2runtimes.py mg_cpu32.json mg.fut:mgNAS "Class B" \
                                    > MG_futhark_cpu32_B.runtimes
../../util/futhark-json2runtimes.py mg_cpu32.json mg.fut:mgNAS "Class C" \
                                    > MG_futhark_cpu32_C.runtimes
../../util/futhark-json2mem.py mg_cpu32.json mg.fut:mgNAS "Class A" \
                                    > MG_futhark_cpu32_A.bytes
../../util/futhark-json2mem.py mg_cpu32.json mg.fut:mgNAS "Class B" \
                                    > MG_futhark_cpu32_B.bytes
../../util/futhark-json2mem.py mg_cpu32.json mg.fut:mgNAS "Class C" \
                                    > MG_futhark_cpu32_C.bytes

futhark bench --backend=multicore mg.fut --json mg_cpu1.json --pass-option=--num-threads=1
../../util/futhark-json2runtimes.py mg_cpu1.json mg.fut:mgNAS "Class A" \
                                    > MG_futhark_cpu1_A.runtimes
../../util/futhark-json2runtimes.py mg_cpu1.json mg.fut:mgNAS "Class B" \
                                    > MG_futhark_cpu1_B.runtimes
../../util/futhark-json2runtimes.py mg_cpu1.json mg.fut:mgNAS "Class C" \
                                    > MG_futhark_cpu1_C.runtimes
../../util/futhark-json2mem.py mg_cpu1.json mg.fut:mgNAS "Class A" \
                                    > MG_futhark_cpu1_A.bytes
../../util/futhark-json2mem.py mg_cpu1.json mg.fut:mgNAS "Class B" \
                                    > MG_futhark_cpu1_B.bytes
../../util/futhark-json2mem.py mg_cpu1.json mg.fut:mgNAS "Class C" \
                                    > MG_futhark_cpu1_C.bytes

futhark bench --backend=cuda mg.fut --json mg_gpu.json
../../util/futhark-json2runtimes.py mg_gpu.json mg.fut:mgNAS "Class A" \
                                    > MG_futhark_gpu_A.runtimes
../../util/futhark-json2runtimes.py mg_gpu.json mg.fut:mgNAS "Class B" \
                                    > MG_futhark_gpu_B.runtimes
../../util/futhark-json2runtimes.py mg_gpu.json mg.fut:mgNAS "Class C" \
                                    > MG_futhark_gpu_C.runtimes
../../util/futhark-json2mem.py mg_gpu.json mg.fut:mgNAS "Class A" \
                                    > MG_futhark_gpu_A.bytes
../../util/futhark-json2mem.py mg_gpu.json mg.fut:mgNAS "Class B" \
                                    > MG_futhark_gpu_B.bytes
../../util/futhark-json2mem.py mg_gpu.json mg.fut:mgNAS "Class C" \
                                    > MG_futhark_gpu_C.bytes
