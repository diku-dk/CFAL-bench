#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_short
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=0:10:00
#SBATCH --output=FlashAttention-gpu.out
#SBATCH --job-name=FlashAttention_baseline-gpu

set -e

rm -f *.runtimes

nvcc -O3 -o flash_attention_gpu flash_attention_gpu.cu -diag-suppress=1650

for sz in 64,16384 64,32768 128,8192 128,16384
do
    IFS=","
    set -- $sz
    printf "\nData sizes are d=$1 and N=$2\n"
    ./flash_attention_gpu $1 $2 \
        | tee FlashAttention_baseline_gpu_d${1}-N${2}.runtimes
done
