set -e

rm -f *.runtimes
rm -f *.memory_kb
rm -f *.compiletimes

cabal build mg

# Measure compilation times
rm -r ~/.cache/accelerate
cabal run mg -- compiletime cpu A > /dev/null
for i in $(seq 1 10);
do
  rm -r ~/.cache/accelerate
  cabal run mg -- compiletime cpu A >> MG_accelerate_cpu_A.compiletimes
done
for i in $(seq 1 10);
do
  rm -r ~/.cache/accelerate
  cabal run mg -- compiletime cpu B >> MG_accelerate_cpu_B.compiletimes
done
for i in $(seq 1 10);
do
  rm -r ~/.cache/accelerate
  cabal run mg -- compiletime cpu C >> MG_accelerate_cpu_C.compiletimes
done
rm -r ~/.cache/accelerate
cabal run mg -- compiletime gpu A > /dev/null
for i in $(seq 1 10);
do
  rm -r ~/.cache/accelerate
  cabal run mg -- compiletime gpu A >> MG_accelerate_gpu_A.compiletimes
done
for i in $(seq 1 10);
do
  rm -r ~/.cache/accelerate
  cabal run mg -- compiletime gpu B >> MG_accelerate_gpu_B.compiletimes
done
for i in $(seq 1 10);
do
  rm -r ~/.cache/accelerate
  cabal run mg -- compiletime gpu C >> MG_accelerate_gpu_C.compiletimes
done

# Measure memory usage
# This only measures the memory usage on the CPU, in kilobytes.
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run mg -- single cpu A
for i in $(seq 1 10);
do
  ACCELERATE_LLVM_NATIVE_THREADS=1 time -v cabal run mg -- single cpu A 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> MG_accelerate_cpu1_A.memory_kb
done
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run mg -- single cpu B
for i in $(seq 1 10);
do
  ACCELERATE_LLVM_NATIVE_THREADS=1 time -v cabal run mg -- single cpu B 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> MG_accelerate_cpu1_B.memory_kb
done
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run mg -- single cpu C
for i in $(seq 1 10);
do
  ACCELERATE_LLVM_NATIVE_THREADS=1 time -v cabal run mg -- single cpu C 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> MG_accelerate_cpu1_C.memory_kb
done

cabal run mg -- single cpu A
for i in $(seq 1 10);
do
  time -v cabal run mg -- single cpu A 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> MG_accelerate_cpu32_A.memory_kb
done
cabal run mg -- single cpu B
for i in $(seq 1 10);
do
  time -v cabal run mg -- single cpu B 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> MG_accelerate_cpu32_B.memory_kb
done
cabal run mg -- single cpu C
for i in $(seq 1 10);
do
  time -v cabal run mg -- single cpu C 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> MG_accelerate_cpu32_C.memory_kb
done

ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run mg -- bench cpu A >> MG_accelerate_cpu1_A.runtimes
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run mg -- bench cpu B >> MG_accelerate_cpu1_B.runtimes
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run mg -- bench cpu C >> MG_accelerate_cpu1_C.runtimes

cabal run mg -- bench cpu A >> MG_accelerate_cpu32_A.runtimes
cabal run mg -- bench cpu B >> MG_accelerate_cpu32_B.runtimes
cabal run mg -- bench cpu C >> MG_accelerate_cpu32_C.runtimes

cabal run mg -- bench gpu A >> MG_accelerate_gpu_A.runtimes
cabal run mg -- bench gpu B >> MG_accelerate_gpu_B.runtimes
cabal run mg -- bench gpu C >> MG_accelerate_gpu_C.runtimes
