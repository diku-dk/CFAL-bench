#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_short
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=0:10:00
#SBATCH --output=flash_attention_dace_cpu_02.out

# module load mkl

# No idea why this is necessary, something
# with slurm and the FPGA
# export XILINX_XRT=/opt/xilinx/xrt


export OMP_PLACES="0:32:1" OMP_PROC_BIND=true

export DACE_library_blas_default_implementation=OpenBLAS
export DACE_compiler_use_cache=1

# OMP_NUM_THREADS=1 python flash_attention_dace.py -d 64 -N 16384
# OMP_NUM_THREADS=1 python flash_attention_dace.py -d 64 -N 32768
# OMP_NUM_THREADS=1 python flash_attention_dace.py -d 128 -N 8192
# OMP_NUM_THREADS=1 python flash_attention_dace.py -d 128 -N 16384

for tile in 32 64 128 256 512
do
    printf "\nTile size is $tile\n"
    for t in 1 2 4 8 16 32
    do
        printf "\nOMP_NUM_THREADS=$t\n"
        printf "Data sizes are d=64 and N=16384\n"
        OMP_NUM_THREADS=$t python flash_attention_dace.py -d 64 -N 16384 -Ti $tile -Tj $tile
        printf "Data sizes are d=64 and N=32768\n"
        OMP_NUM_THREADS=$t python flash_attention_dace.py -d 64 -N 32768 -Ti $tile -Tj $tile
        printf "Data sizes are d=128 and N=8192\n"
        OMP_NUM_THREADS=$t python flash_attention_dace.py -d 128 -N 8192 -Ti $tile -Tj $tile
        printf "Data sizes are d=128 and N=16384\n"
        OMP_NUM_THREADS=$t python flash_attention_dace.py -d 128 -N 16384 -Ti $tile -Tj $tile
    done
done

# OMP_NUM_THREADS=32 python flash_attention_dace.py -d 64 -N 16384
# OMP_NUM_THREADS=32 python flash_attention_dace.py -d 64 -N 32768
# OMP_NUM_THREADS=32 python flash_attention_dace.py -d 128 -N 8192
# OMP_NUM_THREADS=32 python flash_attention_dace.py -d 128 -N 16384

