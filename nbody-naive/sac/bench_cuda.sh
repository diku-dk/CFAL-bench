#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --mem=0
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --output=bench_cuda.out
#SBATCH --gres=gpu:nvidia_a30:1

if [ "$#" -ne 2 ]; then
    printf 'Usage: %s ITER OUTDIR\n' "$0" >&2
    printf '\tITER:   number of times to repeat the experiment\n' >&2
    printf '\tOUTDIR: directory to store result\n' >&2
    exit 1
fi

iter="$1"
outdir="$2"

mkdir -p "$outdir"

make cuda

bench()
{
    n="$1"
    niter="$2"
    name=nbody_cuda_${n}_${niter}

    # Warmup
    {
        i=1
        while [ $i -le "$iter" ]
        do
            ./bin/nbody_cuda "$n" "$niter"
            i=$(( i + 1 ))
        done
    }

    {
        i=1
        while [ $i -le "$iter" ]
        do
            # For cn132
            ./bin/nbody_cuda "$n" "$niter"
            i=$(( i + 1 ))
        done
    } | tee "${outdir}/${name}.raw" | \
    awk '{
               b = a + ($1 - a) / NR;
               q += ($1 - a) * ($1 - b);
               a = b;
             } END {
               printf "%f,%f", a[1], sqrt(q[1] / (NR - 1));
             }' > "${outdir}/${name}.csv"
}

bench 1000 100000
bench 10000 1000
bench 100000 10
