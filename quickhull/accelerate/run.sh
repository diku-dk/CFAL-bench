set -e

rm -f *.runtimes
rm -f *.memory_kb
rm -f *.compiletimes

cabal build quickhull

# Measure compilation times
rm -r ~/.cache/accelerate
cabal run quickhull -- compiletime cpu 100M_rectangle > /dev/null
for i in $(seq 1 10);
do
  rm -r ~/.cache/accelerate
  cabal run quickhull -- compiletime cpu 100M_rectangle >> quickhull_accelerate_cpu.compiletimes
done
rm -r ~/.cache/accelerate
cabal run quickhull -- compiletime gpu 100M_rectangle > /dev/null
for i in $(seq 1 10);
do
  rm -r ~/.cache/accelerate
  cabal run quickhull -- compiletime gpu 100M_rectangle >> quickhull_accelerate_gpu.compiletimes
done

# Measure memory usage
# This only measures the memory usage on the CPU, in kilobytes.
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run quickhull -- single cpu 100M_rectangle
for i in $(seq 1 10);
do
  ACCELERATE_LLVM_NATIVE_THREADS=1 time -v cabal run quickhull -- single cpu 100M_rectangle 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> quickhull_accelerate_cpu1_rectangle.memory_kb
done
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run quickhull -- single cpu 100M_circle
for i in $(seq 1 10);
do
  ACCELERATE_LLVM_NATIVE_THREADS=1 time -v cabal run quickhull -- single cpu 100M_circle 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> quickhull_accelerate_cpu1_circle.memory_kb
done
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run quickhull -- single cpu 100M_quadratic
for i in $(seq 1 10);
do
  ACCELERATE_LLVM_NATIVE_THREADS=1 time -v cabal run quickhull -- single cpu 100M_quadratic 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> quickhull_accelerate_cpu1_quadratic.memory_kb
done

cabal run quickhull -- single cpu 100M_rectangle
for i in $(seq 1 10);
do
  time -v cabal run quickhull -- single cpu 100M_rectangle 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> quickhull_accelerate_cpu32_rectangle.memory_kb
done
cabal run quickhull -- single cpu 100M_circle
for i in $(seq 1 10);
do
  time -v cabal run quickhull -- single cpu 100M_circle 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> quickhull_accelerate_cpu32_circle.memory_kb
done
cabal run quickhull -- single cpu 100M_quadratic
for i in $(seq 1 10);
do
  time -v cabal run quickhull -- single cpu 100M_quadratic 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> quickhull_accelerate_cpu32_quadratic.memory_kb
done

ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run quickhull -- bench cpu 100M_rectangle >> quickhull_accelerate_cpu1_rectangle.runtimes
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run quickhull -- bench cpu 100M_circle >> quickhull_accelerate_cpu1_circle.runtimes
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run quickhull -- bench cpu 100M_quadratic >> quickhull_accelerate_cpu1_quadratic.runtimes

cabal run quickhull -- bench cpu 100M_rectangle >> quickhull_accelerate_cpu32_rectangle.runtimes
cabal run quickhull -- bench cpu 100M_circle >> quickhull_accelerate_cpu32_circle.runtimes
cabal run quickhull -- bench cpu 100M_quadratic >> quickhull_accelerate_cpu32_quadratic.runtimes

cabal run quickhull -- bench gpu 100M_rectangle >> quickhull_accelerate_gpu_rectangle.runtimes
cabal run quickhull -- bench gpu 100M_circle >> quickhull_accelerate_gpu_circle.runtimes
cabal run quickhull -- bench gpu 100M_quadratic >> quickhull_accelerate_gpu_quadratic.runtimes
