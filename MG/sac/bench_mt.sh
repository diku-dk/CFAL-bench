#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=mt.out

# No idea why this is necessary, something
# with slurm and the FPGA
export XILINX_XRT=/opt/xilinx/xrt

if [ "$#" -ne 4 ]; then
    printf 'Usage: run.sh N CLASS RUNS OUT_DIR P\n\n' >&2
    printf '\tCLASS: Problem class (S, W, A, B, C, D) \n\n' >&2
    printf '\tRUNS: How often to run the benchmark\n\n' >&2
    printf '\tOUT_DIR: Directory to store benchmark results\n\n' >&2
    printf '\tP: Number of processors to use\n\n' >&2
    exit 1
fi

class="$1"
runs="$2"
outfile="$3/MG_${class}_mt_sac"
pmax="$4"
mkdir -p "$3"

make CLASS="$class" mt -j

printf 'p,mean,stddev\n' > "${outfile}"

p=1
while [ $p -le "$pmax" ]
do
    printf '%d,' "$p" >> "${outfile}"
    {
        i=1
        while [ $i -le "$runs" ]
        do
            "bin/MG_${class}_mt" -mt "$p"
            i=$(( i + 1 ))
        done
    } | variance >> "${outfile}"
    p=$(( 2 * p ))
done
