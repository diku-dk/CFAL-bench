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

printf "%15s  Acc  APL DaCe  Fut  SAC\n" ""

printf "%15s %4d %4d %4d %4d %4d\n" \
       nbody \
       $(sloc_accelerate nbody-naive/accelerate/nbody-naive/src/*.hs) \
       $(sloc_apl nbody-naive/APL/nbody_naive.apln) \
       $(sloc_dace nbody-naive/dace/nbody_dace.py) \
       $(sloc_futhark nbody-naive/futhark/nbody.fut) \
       $(sloc_sac nbody-naive/sac/src/nbody.sac)

printf "%15s %4d %4d %4d %4d %4d\n" \
       LocVolCalib \
       $(sloc_accelerate LocVolCalib/accelerate/src/*.hs) \
       $(sloc_apl LocVolCalib/APL/finpar.apln) \
       $(sloc_dace LocVolCalib/dace/LocVolCalib.py) \
       $(sloc_futhark LocVolCalib/futhark/LocVolCalib.fut) \
       $(sloc_sac LocVolCalib/sac/src/VolCalibGPU.sac)

printf "%15s %4d %4d %4d %4d %4d\n" \
       MG \
       $(sloc_accelerate MG/accelerate/src/*.hs) \
       $(sloc_apl MG/APL/MG.apln) \
       $(sloc_dace MG/dace/mg.py) \
       $(sloc_futhark MG/futhark/mg.fut) \
       $(sloc_sac MG/sac/src/*.sac)

printf "%15s %4d %4d %4d %4d %4d\n" \
       QuickHull \
       $(sloc_accelerate quickhull/accelerate/src/*.hs) \
       $(sloc_apl quickhull/APL/quickhull.apln) \
       $(sloc_dace quickhull/dace/quickhull.py) \
       $(sloc_futhark quickhull/futhark/*.fut) \
       $(sloc_sac quickhull/sac/src/*.sac)

printf "%15s %4d %4d %4d %4d %4d\n" \
       FlashAttention \
       $(sloc_accelerate FlashAttention/accelerate/src/*.hs) \
       $(sloc_apl FlashAttention/APL/flash_attention.apln) \
       $(sloc_dace FlashAttention/dace/flash_attention_dace.py) \
       $(sloc_futhark FlashAttention/futhark/custom-alg1-opt.fut) \
       $(sloc_sac FlashAttention/sac/src/*.sac)
