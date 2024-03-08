#!/bin/sh

if [ "$#" -ne 3 ]; then
	printf "Usage: test.sh d N M\n\n" >&2
	printf "\td:    Number of columns of the matrices.\n" >&2
	printf "\tN:    Number of rows of the matrices.\n" >&2
    printf "\tM:    Size of blocks in bytes (recommended: half L2).\n" >&2
	exit 1
fi

d="$1"
N="$2"
M="$3"

inputf=$(mktemp)
attentiono=$(mktemp)
flasho=$(mktemp)
alg1o=$(mktemp)
blas_alg1o=$(mktemp)
custom_alg1o=$(mktemp)
custom_alg1o_mt=$(mktemp)

make -j

bin/generate "$d" "$N" -random > "$inputf"

OMP_NUM_THREADS=1 bin/attention -io < "$inputf" > "$attentiono"

printf "Attention vs flash\n"
OMP_NUM_THREADS=1 bin/flash -io < "$inputf" > "$flasho"
bin/compare "$d" "$M" "$attentiono" "$flasho"

printf "Attention vs flash_alg1\n"
bin/flash_alg1 "$M" -io < "$inputf" > "$alg1o"
bin/compare "$d" "$M" "$attentiono" "$alg1o"

printf "Attention vs blas_alg1\n"
OMP_NUM_THREADS=1 bin/blas_alg1 "$M" -io < "$inputf" > "$blas_alg1o"
bin/compare "$d" "$M" "$attentiono" "$blas_alg1o"

printf "Attention vs custom_alg1\n"
bin/custom_alg1 "$M" -io < "$inputf" > "$custom_alg1o"
bin/compare "$d" "$M" "$attentiono" "$custom_alg1o"

printf "Attention vs custom_alg1_mt\n"
OMP_NUM_THREADS=4 bin/custom_alg1_mt "$M" -io < "$inputf" > "$custom_alg1o_mt"
bin/compare "$d" "$M" "$attentiono" "$custom_alg1o_mt"
