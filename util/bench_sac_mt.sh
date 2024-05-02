#!/bin/bash

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_short
#SBATCH --cpus-per-task=32
#SBATCH --output=bench_omp.out

# If the script is used improperly, prints a short explanation
# and exits
if [ "$#" -lt 5 ]; then
    printf 'Usage: bench_sac.sh BINARY RUNS OUT_DIR OUT_NAME P ARGS\n\n' >&2
    printf '\tBINARY: SaC program to benchmark\n' >&2
    printf '\t        Assumes only GFLOPS/s is printed to stdout\n\n' >&2
    printf '\tRUNS: How often to run the benchmark\n\n' >&2
    printf '\tOUT_DIR: Directory to store benchmark results\n\n' >&2
    printf '\tOUT_NAME: Name of benchmark csv\n\n' >&2
    printf '\tP: Maximum number of cores\n\n' >&2
    printf '\tARGS: Commandline arguments for the binary\n\n' >&2
    exit 1
fi

# Parse commandline arguments (TODO get rid of bash-specific shift)
binary="$1"                         # Program to benchmark
shift
runs="$1"                           # How often to repeat each experiment
shift
outdir="$1"
shift
outname="$1"
shift
pmax="$1"                           # Will experiment on 1, 2, 4, ..., pmax
                                    # processors
shift
args=$(printf '%s ' "$@")           # Arguments for binary
outfile="${outdir}/${outname}"      # File to write the results to

mkdir -p "$outdir"                  # Creates the directory for the outfile if
                                    # it does not yet exist

printf 'p,mean,stddev\n' > "${outfile}"  # Create header of csv file

p=1
while [ $p -le "$pmax" ] # This loop iterates over 1, 2, 4, ..., pmax processors
do
    printf '%d,' "$p" >> "${outfile}" # Print number of processors
    {
        i=1          # This while loop repeats the experiment "$runs" times.
                     # The { } block around this groups the stdout of this loop.
                     # So for example if "$runs" = 4, and the Gflops/s are
                     # around 1.0, you could get something like
                     # "1.0 0.94 1.03 1.02"
        while [ $i -le "$runs" ]
        do
            # OMP_NUM_THREADS sets the number of cores to use
            "$binary" $args -mt "$p" -mt_bind simple
            i=$(( i + 1 ))
        done
    } | variance >> "${outfile}" # Take the grouped stdout and pipe it into
                                 # a program called 'variance'
                                 # This will compute the mean and standard
                                 # deviation, outputting it on stdout.
                                 # The >> "${outfile}" then redirects stdout
                                 # to outfile
    p=$(( 2 * p ))
done
