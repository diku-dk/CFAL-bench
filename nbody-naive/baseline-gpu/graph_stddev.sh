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

if [ "$#" -ne 4 ]; then
    printf 'Usage: %s N ITERATIONS RUNS OUT_DIR\n\n' "$0" >&2
    printf '\tN: Number of bodies\n\n' >&2
    printf '\tITERATIONS: Number of advancements in the benchmark\n\n' >&2
    printf '\tRUNS: How often to run the benchmark\n\n' >&2
    printf '\tOUT_DIR: Directory to store benchmark results\n\n' >&2
    exit 1
fi

n="$1"
iter="$2"
runs="$3"
outfile="$4/nbody_${n}_${iter}_stddev"
mkdir -p "$4"

make

printf 'n,stddev\n' > "${outfile}"

./nbody "$n" "$iter" "$runs" | sed 's/Baseline\ (GPU),n=[^,]*,//g' |
awk '{
           b = a + ($1 - a) / NR;
           q += ($1 - a) * ($1 - b);
           a = b;
           printf "%d,%f\n", NR, sqrt(q / NR);
         }' >> "${outfile}"
