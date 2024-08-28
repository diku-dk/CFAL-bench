#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_short
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=0:10:00
#SBATCH --output=flash_attention_dace_gpu_new.out

# export OMP_PLACES=cores
export OMP_PLACES="0:32:1"
export OMP_PROC_BIND=true

python bench_flash_attention_dace_gpu.py

# export DACE_library_blas_default_implementation=MKL
# export DACE_library_blas_default_implementation=OpenBLAS
# export DACE_compiler_use_cache=1


# export DACE_library_blas_default_implementation=OpenBLAS
# OMP_NUM_THREADS=1 python bench_flash_attention_dace_cpu.py

# OMP_NUM_THREADS=32 python flash_attention_dace.py -d 64 -N 16384 -Ti 512 -Tj 512
# OMP_NUM_THREADS=32 python flash_attention_dace.py -d 64 -N 32768
# OMP_NUM_THREADS=32 python flash_attention_dace.py -d 128 -N 8192
# OMP_NUM_THREADS=32 python flash_attention_dace.py -d 128 -N 16384

