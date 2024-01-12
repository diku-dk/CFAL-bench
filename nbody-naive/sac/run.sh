#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=run.out

# No idea why this is necessary, something
# with slurm and the FPGA
export XILINX_XRT=/opt/xilinx/xrt

make clean
make -j3

echo "Sequential"
bin/nbody_seq
echo "MT"
bin/nbody_mt -mt 32
echo "Cuda"
bin/nbody_cuda
