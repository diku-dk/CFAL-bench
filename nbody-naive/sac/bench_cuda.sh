#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --mem=0
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --output=bench_mt.out

if [ "$#" -ne 2 ]; then
    printf 'Usage: %s ITER OUTDIR\n' "$0" >&2
    printf '\tITER:   number of times to repeat the experiment\n' >&2
    printf '\tOUTDIR: directory to store result\n' >&2
    exit 1
fi

iter="$1"
outdir="$2"

mkdir -p "$outdir"

bench()
{
    n="$1"
    niter="$2"
    {
        i=1
        while [ $i -le "$iter" ]
        do
            # For cn132
            numactl --interleave all ./bin/nbody_cuda "$n" "$niter"
            i=$(( i + 1 ))
        done
    } | awk '{
               for (i = 1; i <= NF; i++) {
                   b[i] = a[i] + ($i - a[i]) / NR;
                   q[i] += ($i - a[i]) * ($i - b[i]);
                   a[i] = b[i];
               }
             } END {
               printf "%f,%f", a[1], sqrt(q[1] / NR);
               for (i = 2; i <= NF; i++) {
                   printf ",%f,%f", a[i], sqrt(q[i] / NR);
               }
               print "";
             }' > "${outdir}/nbody_cuda_${n}_${niter}.csv"
}

bench 1000 100000
bench 10000 1000
bench 100000 10
