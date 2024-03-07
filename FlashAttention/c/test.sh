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

make -j

bin/generate "$d" "$N" -random > "$inputf"
bin/attention -io < "$inputf" > "$attentiono"

bin/flash -io < "$inputf" > "$flasho"
bin/flash_alg1 "$M" -io < "$inputf" > "$alg1o"
bin/blas_alg1 "$M" -io < "$inputf" > "$blas_alg1o"

printf "Attention vs flash\n"
bin/compare "$d" "$M" "$attentiono" "$flasho"

printf "Attention vs flash_alg1\n"
bin/compare "$d" "$M" "$attentiono" "$alg1o"

printf "Attention vs blas_alg1\n"
bin/compare "$d" "$M" "$attentiono" "$blas_alg1o"
