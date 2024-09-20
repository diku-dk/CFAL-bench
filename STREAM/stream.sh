#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --mem=0
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --output=stream_cn132.out

make clean
make stream_c
OMP_PLACES=cores numactl --interleave all ./stream_c
