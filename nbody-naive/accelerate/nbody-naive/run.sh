set -e

rm -f *.runtimes
rm -f *.memory_kb
rm -f *.compiletimes

cabal build nbody-naive

# Measure compilation times
cabal run nbody-naive -- compiletime cpu n1000 > /dev/null
for i in $(seq 1 10);
do
  rm -r ~/.cache/accelerate
  cabal run nbody-naive -- compiletime cpu n1000 >> nbody_accelerate_cpu.compiletimes
done
cabal run nbody-naive -- compiletime gpu n1000 > /dev/null
for i in $(seq 1 10);
do
  rm -r ~/.cache/accelerate
  cabal run nbody-naive -- compiletime gpu n1000 >> nbody_accelerate_gpu.compiletimes
done

# Measure memory usage
# This only measures the memory usage on the CPU, in kilobytes.
for i in $(seq 1 10);
do
  ACCELERATE_LLVM_NATIVE_THREADS=1 time -v cabal run nbody-naive -- single cpu n1000 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> nbody_accelerate_cpu1_n1000.memory_kb
done
for i in $(seq 1 10);
do
  ACCELERATE_LLVM_NATIVE_THREADS=1 time -v cabal run nbody-naive -- single cpu n10000 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> nbody_accelerate_cpu1_n10000.memory_kb
done
for i in $(seq 1 10);
do
  ACCELERATE_LLVM_NATIVE_THREADS=1 time -v cabal run nbody-naive -- single cpu n100000 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> nbody_accelerate_cpu1_n100000.memory_kb
done

for i in $(seq 1 10);
do
  time -v cabal run nbody-naive -- single cpu n1000 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> nbody_accelerate_cpu32_n1000.memory_kb
done
for i in $(seq 1 10);
do
  time -v cabal run nbody-naive -- single cpu n10000 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> nbody_accelerate_cpu32_n10000.memory_kb
done
for i in $(seq 1 10);
do
  time -v cabal run nbody-naive -- single cpu n100000 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> nbody_accelerate_cpu32_n100000.memory_kb
done

ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run nbody-naive -- bench cpu n1000 >> nbody_accelerate_cpu1_n1000.runtimes
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run nbody-naive -- bench cpu n10000 >> nbody_accelerate_cpu1_n10000.runtimes
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run nbody-naive -- bench cpu n100000 >> nbody_accelerate_cpu1_n100000.runtimes

cabal run nbody-naive -- bench cpu n1000 >> nbody_accelerate_cpu32_n1000.runtimes
cabal run nbody-naive -- bench cpu n10000 >> nbody_accelerate_cpu32_n10000.runtimes
cabal run nbody-naive -- bench cpu n100000 >> nbody_accelerate_cpu32_n100000.runtimes

cabal run nbody-naive -- bench gpu n1000 >> nbody_accelerate_gpu_n1000.runtimes
cabal run nbody-naive -- bench gpu n10000 >> nbody_accelerate_gpu_n10000.runtimes
cabal run nbody-naive -- bench gpu n100000 >> nbody_accelerate_gpu_n100000.runtimes
