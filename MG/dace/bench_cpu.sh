#!/bin/sh

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_long
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=mg_dace_cpu_new_01T.out

# No idea why this is necessary, something
# with slurm and the FPGA
export XILINX_XRT=/opt/xilinx/xrt

# export DACE_profiling=1
# export DACE_treps=10
export DACE_compiler_use_cache=0
export OMP_PLACES="0:32:1"
export OMP_PROC_BIND=true

# echo "Running MG-CPU on dace on 1T"
# export OMP_NUM_THREADS=1
# python mg.py -c A -f dace_cpu
# python mg.py -c B -f dace_cpu
# python mg.py -c C -f dace_cpu

echo "Running MG-CPU on dace on 32T"
export OMP_NUM_THREADS=32
# python mg.py -c A -f dace_cpu
# python mg.py -c B -f dace_cpu
python mg.py -c C -f dace_cpu

