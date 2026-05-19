set -e

rm -f *.runtimes
rm -f *.compiletimes

cabal build nbody-naive

# Measure compilation times
cabal run nbody-naive -- compiletime cpu n1000 >> nbody_accelerate_cpu.compiletimes
cabal run nbody-naive -- compiletime gpu n1000 >> nbody_accelerate_gpu.compiletimes

# Measure memory usage
# This only measures the memory usage on the CPU.
# The numbers are not automatically extracted, but is reported after
# "Maximum resident set size (kbytes)"
time -v cabal run nbody-naive -- single cpu n1000
time -v cabal run nbody-naive -- single cpu n10000
time -v cabal run nbody-naive -- single cpu n100000

ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run nbody-naive -- bench cpu n1000 >> nbody_accelerate_cpu1_n1000.runtimes
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run nbody-naive -- bench cpu n10000 >> nbody_accelerate_cpu1_n10000.runtimes
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run nbody-naive -- bench cpu n100000 >> nbody_accelerate_cpu1_n100000.runtimes

cabal run nbody-naive -- bench cpu n1000 >> nbody_accelerate_cpu32_n1000.runtimes
cabal run nbody-naive -- bench cpu n10000 >> nbody_accelerate_cpu32_n10000.runtimes
cabal run nbody-naive -- bench cpu n100000 >> nbody_accelerate_cpu32_n100000.runtimes

cabal run nbody-naive -- bench gpu n1000 >> nbody_accelerate_gpu_n1000.runtimes
cabal run nbody-naive -- bench gpu n10000 >> nbody_accelerate_gpu_n10000.runtimes
cabal run nbody-naive -- bench gpu n100000 >> nbody_accelerate_gpu_n100000.runtimes
