#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_short
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=0:10:00
#SBATCH --output=flash_attention_cpu.out

export OMP_PLACES="0:32:1"
export OMP_PROC_BIND=true

make clean && make

for sz in 64,16384 64,32768 128,8192 128,16384
do
    IFS=","
    set -- $sz
    printf "\nData sizes are d=$1 and N=$2\n"
    for M in 64 128 256 512 1024 2048
    do
        printf "\nM=$M\n"
    for t in 1 32
        do
            printf "OMP_NUM_THREADS=$t\n"
            OMP_NUM_THREADS=$t bin/flash_attention_cpu $1 $2 $M
        done
    done
done
