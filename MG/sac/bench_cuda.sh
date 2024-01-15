#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=cuda.out

# No idea why this is necessary, something
# with slurm and the FPGA
export XILINX_XRT=/opt/xilinx/xrt

if [ "$#" -ne 3 ]; then
    printf 'Usage: run.sh CLASS RUNS OUT_DIR\n\n' >&2
    printf '\tCLASS: Problem class (S, A, B, C, D)\n\n' >&2
    printf '\tRUNS: How often to run the benchmark\n\n' >&2
    printf '\tOUT_DIR: Directory to store benchmark results.\n\n' >&2
    exit 1
fi

class="$1"
runs="$2"
outfile="$3/MG_${class}_cuda_sac"
mkdir -p "$3"

make CLASS="$class" cuda

{
printf 'mean,stddev\n'
} > "$outfile"

i=1
while [ $i -le 5 ]
do
    bin/MG_"${class}"_cuda
    i=$(( i + 1 ))
done

i=1
{
while [ $i -le "$runs" ]
do
    bin/MG_"${class}"_cuda
    i=$(( i + 1 ))
done
} | variance >> "$outfile"
