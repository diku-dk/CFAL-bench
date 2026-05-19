#!/usr/bin/env bash
#SBATCH --partition=csmpi_fpga_long
#SBATCH --job-name=cfal-accelerate
#SBATCH --time=10:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G

export PATH="/vol/itt/data/cfal/haskell/.ghcup/bin:/vol/itt/data/cfal/llvm/LLVM-21.1.8-Linux-X64/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64"
export CABAL_DIR="/vol/itt/data/cfal/accelerate-shared-build-workdir/gkeller/cabal-dir"

sh run.sh
