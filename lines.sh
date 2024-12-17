#!/bin/sh
#
# This script counts source lines of code (ignoring comments and
# blanks) for the different benchmark implementations.

set -e

sloc_accelerate() {
    grep -E -x -v '\w*(--.*)?' -- "$@" | wc -l
}

sloc_apl() {
    grep -E -x -v '\w*(⍝.*)?' -- "$@" | wc -l
}

sloc_dace() {
    grep -E -x -v '\w*(#.*)?' -- "$@" | wc -l
}

sloc_futhark() {
    grep -E -x -v '\w*(--.*)?' -- "$@" | wc -l
}

sloc_sac() {
    grep -E -x -v '\w*(//.*)?' -- "$@" | wc -l
}

sloc_cpp() {
    scc -f csv -i c,cpp,h,cl,cu "$@"|tail -n +2|cut -d, -f2|awk '{sum+=$1}END{print sum}'
}

printf "%15s                                baselines\n" ""
printf "%15s  Acc  APL DaCe--C/G  Fut  SAC  CPU   GPU\n" ""

printf "%15s %4d %4d %4d %4d %4d %4d %4d %5d\n" \
       nbody \
       $(sloc_accelerate nbody-naive/accelerate/nbody-naive/src/*.hs) \
       $(sloc_apl nbody-naive/APL/nbody_naive_v4.apln) \
       $(sloc_dace nbody-naive/dace/nbody_dace_cpu_impl.py) \
       $(sloc_dace nbody-naive/dace/nbody_dace_gpu_impl.py) \
       $(sloc_futhark nbody-naive/futhark/nbody.fut) \
       $(sloc_sac nbody-naive/sac/src/nbody.sac) \
       $(sloc_cpp nbody-naive/baseline-cpu) \
       $(sloc_cpp nbody-naive/baseline-gpu)

printf "%15s %4d %4d %4d %4s %4d %4d %4d %5d\n" \
       LocVolCalib \
       $(sloc_accelerate LocVolCalib/accelerate/src/*.hs) \
       $(sloc_apl LocVolCalib/APL/finpar.apln) \
       $(sloc_dace LocVolCalib/dace/locvolcalib_dace.py) \
       "same" \
       $(sloc_futhark LocVolCalib/futhark/LocVolCalib.fut) \
       $(sloc_sac LocVolCalib/sac/src/VolCalibGPU.sac) \
       $(sloc_cpp LocVolCalib/baseline-FinPar/COpenMP) \
       $(sloc_cpp LocVolCalib/baseline-FinPar/AllParOpenCLMP)

printf "%15s %4d %4d %4d %4s %4d %4d %4s %5d\n" \
       MG \
       $(sloc_accelerate MG/accelerate/src/*.hs) \
       $(sloc_apl MG/APL/MG_k.apln) \
       $(sloc_dace MG/dace/mg_dace.py) \
       "same" \
       $(sloc_futhark MG/futhark/mg.fut) \
       $(sloc_sac MG/sac/src/*.sac) \
       "---" \
       $(sloc_cpp MG/baseline-NPB-GPU/MG)


printf "%15s %4d %4d %4s %4s %4d %4d %4d %5s\n" \
       QuickHull \
       $(sloc_accelerate quickhull/accelerate/src/*.hs) \
       $(sloc_apl quickhull/APL/quickhull_k.apln) \
       "---" \
       "---" \
       $(sloc_futhark quickhull/futhark/*.fut) \
       $(sloc_sac quickhull/sac/src/*.sac) \
       $(sloc_cpp quickhull/baseline/hull.cpp) \
       "---"

printf "%15s %4d %4d %4d %4d %4d %4d %4d %5d\n" \
       FlashAttention \
       $(sloc_accelerate FlashAttention/accelerate/src/*.hs) \
       $(sloc_apl FlashAttention/APL/flash_attention_k.apln) \
       $(sloc_dace FlashAttention/dace/flash_attention_dace_cpu.py) \
       $(sloc_dace FlashAttention/dace/flash_attention_dace_gpu.py) \
       $(sloc_futhark FlashAttention/futhark/custom-alg1-opt.fut) \
       $(sloc_sac FlashAttention/sac/src/*.sac) \
       $(sloc_cpp FlashAttention/baseline-cpu) \
       $(sloc_cpp FlashAttention/baseline-gpu)
