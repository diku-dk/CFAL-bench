set -e

rm -f *.runtimes
rm -f *.memory_kb
rm -f *.compiletimes

cabal build flashattention

# Measure compilation times
rm -r ~/.cache/accelerate
cabal run flashattention -- compiletime cpu d64-N16384 > /dev/null
for i in $(seq 1 10);
do
  rm -r ~/.cache/accelerate
  cabal run flashattention -- compiletime cpu d64-N16384 >> FlashAttention_accelerate_cpu.compiletimes
done
rm -r ~/.cache/accelerate
cabal run flashattention -- compiletime gpu d64-N16384 > /dev/null
for i in $(seq 1 10);
do
  rm -r ~/.cache/accelerate
  cabal run flashattention -- compiletime gpu d64-N16384 >> FlashAttention_accelerate_gpu.compiletimes
done

# Measure memory usage
# This only measures the memory usage on the CPU, in kilobytes.
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run flashattention -- single cpu d64-N16384
for i in $(seq 1 10);
do
  ACCELERATE_LLVM_NATIVE_THREADS=1 time -v cabal run flashattention -- single cpu d64-N16384 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> FlashAttention_accelerate_cpu1_d64-N16384.memory_kb
done
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run flashattention -- single cpu d64-N32768
for i in $(seq 1 10);
do
  ACCELERATE_LLVM_NATIVE_THREADS=1 time -v cabal run flashattention -- single cpu d64-N32768 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> FlashAttention_accelerate_cpu1_d64-N32768.memory_kb
done
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run flashattention -- single cpu d128-N8192
for i in $(seq 1 10);
do
  ACCELERATE_LLVM_NATIVE_THREADS=1 time -v cabal run flashattention -- single cpu d128-N8192 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> FlashAttention_accelerate_cpu1_d128-N8192.memory_kb
done
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run flashattention -- single cpu d128-N16384
for i in $(seq 1 10);
do
  ACCELERATE_LLVM_NATIVE_THREADS=1 time -v cabal run flashattention -- single cpu d128-N16384 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> FlashAttention_accelerate_cpu1_d128-N16384.memory_kb
done

cabal run flashattention -- single cpu d64-N16384
for i in $(seq 1 10);
do
  time -v cabal run flashattention -- single cpu d64-N16384 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> FlashAttention_accelerate_cpu32_d64-N16384.memory_kb
done
cabal run flashattention -- single cpu d64-N32768
for i in $(seq 1 10);
do
  time -v cabal run flashattention -- single cpu d64-N32768 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> FlashAttention_accelerate_cpu32_d64-N32768.memory_kb
done
cabal run flashattention -- single cpu d128-N8192
for i in $(seq 1 10);
do
  time -v cabal run flashattention -- single cpu d128-N8192 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> FlashAttention_accelerate_cpu32_d128-N8192.memory_kb
done
cabal run flashattention -- single cpu d128-N16384
for i in $(seq 1 10);
do
  time -v cabal run flashattention -- single cpu d128-N16384 2>&1 | sed -n 's/.*Maximum resident set size (kbytes): //p' >> FlashAttention_accelerate_cpu32_d128-N16384.memory_kb
done

ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run flashattention -- bench cpu d64-N16384 >> FlashAttention_accelerate_cpu1_d64-N16384.runtimes
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run flashattention -- bench cpu d64-N32768 >> FlashAttention_accelerate_cpu1_d64-N32768.runtimes
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run flashattention -- bench cpu d128-N8192 >> FlashAttention_accelerate_cpu1_d128-N8192.runtimes
ACCELERATE_LLVM_NATIVE_THREADS=1 cabal run flashattention -- bench cpu d128-N16384 >> FlashAttention_accelerate_cpu1_d128-N16384.runtimes

cabal run flashattention -- bench cpu d64-N16384 >> FlashAttention_accelerate_cpu32_d64-N16384.runtimes
cabal run flashattention -- bench cpu d64-N32768 >> FlashAttention_accelerate_cpu32_d64-N32768.runtimes
cabal run flashattention -- bench cpu d128-N8192 >> FlashAttention_accelerate_cpu32_d128-N8192.runtimes
cabal run flashattention -- bench cpu d128-N16384 >> FlashAttention_accelerate_cpu32_d128-N16384.runtimes

cabal run flashattention -- bench gpu d64-N16384 >> FlashAttention_accelerate_gpu_d64-N16384.runtimes
cabal run flashattention -- bench gpu d64-N32768 >> FlashAttention_accelerate_gpu_d64-N32768.runtimes
cabal run flashattention -- bench gpu d128-N8192 >> FlashAttention_accelerate_gpu_d128-N8192.runtimes
cabal run flashattention -- bench gpu d128-N16384 >> FlashAttention_accelerate_gpu_d128-N16384.runtimes
