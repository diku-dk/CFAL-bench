#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=omp.out

# No idea why this is necessary, something
# with slurm and the FPGA
export XILINX_XRT=/opt/xilinx/xrt

if [ "$#" -ne 5 ]; then
    printf 'Usage: run.sh N ITERATIONS RUNS OUT_DIR P\n\n' >&2
    printf '\tN: Number of bodies\n\n' >&2
    printf '\tITERATIONS: Number of advancements in the benchmark\n\n' >&2
    printf '\tRUNS: How often to run the benchmark\n\n' >&2
    printf '\tOUT_DIR: Directory to store benchmark results\n\n' >&2
    printf '\tP: Number of processors to use\n\n' >&2
    exit 1
fi

n="$1"
iter="$2"
runs="$3"
outfile="$4/nbody_${n}_${iter}_omp_C"
pmax="$5"
mkdir -p "$4"

make clean
make -j

printf 'p,mean,stddev\n' > "${outfile}"

p=1
while [ $p -le "$pmax" ]
do
    printf '%d' "$p" >> "${outfile}"
    {
        i=1
        while [ $i -le "$runs" ]
        do
            OMP_NUM_THREADS="$p" ./nbody_omp "$n" "$iter"
            i=$(( i + 1 ))
        done
    } | variance >> "${outfile}"
    p=$(( 2 * p ))
done
