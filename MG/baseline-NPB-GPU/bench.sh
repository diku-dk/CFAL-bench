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
    printf 'Usage: bench_omp.sh CLASS RUNS OUT_DIR P\n\n' >&2
    printf '\tCLASS: Problem class (S, W, A, B, C, D) \n\n' >&2
    printf '\tRUNS: How often to run the benchmark\n\n' >&2
    printf '\tOUT_DIR: Directory to store benchmark results\n\n' >&2
    exit 1
fi

class="$1"
runs="$2"
outfile="$3/MG_${class}_cuda"
mkdir -p "$3"

make clean
make mg CLASS="$class" 

printf 'mean,stddev\n' > "${outfile}"

i=1
while [ $i -le 5 ]
do
    "bin/mg.${class}"
    i=$(( i + 1 ))
done

i=1
{
    while [ $i -le "$runs" ]
    do
        "bin/mg.${class}"
        i=$(( i + 1 ))
    done
} | grep Mop\/s\ total | grep -o '[0-9.]*' | \
    awk '{print $1/1000}' /dev/stdin | variance >> "$outfile"
