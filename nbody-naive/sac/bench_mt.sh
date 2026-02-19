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

if [ "$#" -ne 2 ]; then
    printf 'Usage: %s RUNS OUT_DIR\n\n' "$0" >&2
    exit 1
fi

runs="$1"
outdir="$2"

make mt
mkdir -p "$outdir"

bench()
{
    n="$1"
    iter="$2"

    name=nbody_mt_${n}_${iter}

    # Warmup
    {
        i=1
        while [ $i -le 3 ]
        do
            SAC_PARALLEL=32 /usr/bin/time -v numactl --interleave all \
                ./bin/nbody_mt "$n" "$iter"
            i=$(( i + 1 ))
        done
    }

    {
        {
            i=1
            while [ $i -le "$runs" ]
            do
                SAC_PARALLEL=32 /usr/bin/time -v numactl --interleave all \
                    ./bin/nbody_mt "$n" "$iter"
                i=$(( i + 1 ))
            done
        } | tee "${outdir}/${name}.raw" | \
        awk '{
                   b = a + ($1 - a) / NR;
                   q += ($1 - a) * ($1 - b);
                   a = b;
                 } END {
                   printf "%f,%f", a, sqrt(q / (NR - 1));
                 }' > "${outdir}/${name}.csv"
    } 2>&1 | \
      grep "Maximum resident" | \
      sed  's/^[^:]*:[ ]*//g' | \
      awk '{print $1 * 1000}' | \
      tee "${outdir}/${name}_mem.raw" | \
      awk '{
               b = a + ($1 - a) / NR;
               q += ($1 - a) * ($1 - b);
               a = b;
             } END {
               printf "%f,%f", a, sqrt(q / (NR - 1));
             }' > "${outdir}/${name}_mem.csv"
}

bench 1000 100000
bench 10000 1000
bench 100000 10
