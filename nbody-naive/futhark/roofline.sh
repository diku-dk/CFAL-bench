#!/bin/sh
#
# Job for computing ncu data. Somewhat manual, but it works.

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_short
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G
#SBATCH --time=0:10:00
#SBATCH --output=nbody_futhark_roofline.out
#SBATCH --job-name=nbody_fut_roofline

# Ensure that the necessary tools are in PATH.
export PATH=/vol/itt/data/cfal/team-futhark/bin/:$PATH

# Fix a CUDA version.
CUDA=/usr/local/cuda
export LIBRARY_PATH=$CUDA/lib64:$CUDA/lib64/stubs
export LD_LIBRARY_PATH=$CUDA/lib64/
export CPATH=$CUDA/include

export CFLAGS='-Ofast -march=native -mtune=native'

set -e

futhark c nbody.fut --server
futhark script ./nbody -e 'mk_positions 1000' > 1000.positions
futhark script ./nbody -e 'mk_masses 1000' > 1000.masses
futhark script ./nbody -e 'mk_positions 10000' > 10000.positions
futhark script ./nbody -e 'mk_masses 10000' > 10000.masses
futhark script ./nbody -e 'mk_positions 100000' > 100000.positions
futhark script ./nbody -e 'mk_masses 100000' > 100000.masses
futhark cuda nbody.fut --executable
(echo 1i32 0.1f64; cat 1000.positions 1000.masses) > 1000.input
(echo 1i32 0.1f64; cat 10000.positions 10000.masses) > 10000.input
(echo 1i32 0.1f64; cat 100000.positions 100000.masses) > 100000.input
ncu --set full -f --export futhark_roofline_1000_1.ncu-rep ./nbody -n < 1000.input
ncu --set full -f --export futhark_roofline_10000_1.ncu-rep ./nbody -n < 10000.input
ncu --set full -f --export futhark_roofline_100000_1.ncu-rep ./nbody -n < 100000.input
