# How to run the benchmarks

This document is meant to contain instructions for how to run the
various benchmarks on the shared system, including if necessary how to
install necessary software.

Eventually I would also like to automate this with a script.

## Generally

First, follow Sven-Bodo's instructions on how to log in and make your
way to the cnlogin22 node.

You will need to use slurm to enqueue jobs. When enqueuing, you need
to provide a time limit, a partition, a core count, and some resources
(specifically, whether you want a GPU). The partition is mandatory,
but the others have defaults. I find that the `csmpi_fpga_short`
partition is the most useful (allows jobs up to 30 minutes). This runs
a command (`ls`) with a ten minute timeout on `csmpi_fpga_short`:

```
$ srun -t 10:00 -p csmpi_fpga_short ls
```

I can also run `ls` with 32 cores (seemingly the capacity of the nodes
in the `csmpi_fpga_short` partition):

```
$ srun -t 10:00 -p csmpi_fpga_short -c 32 ls
```

And perhaps more interestingly, I can ask to be allocated a GPU,
listing my files at breakneck speed:

```
$ srun -t 10:00 -p csmpi_fpga_short --gres=gpu:nvidia_a30:1 ls
```

For example, this is how I benchmark the Futhark implementation of the
N-body benchmark:

```
$ srun -t 10:00 -p csmpi_fpga_short --gres=gpu:nvidia_a30:1 futhark bench --backend=cuda .
srun: job 4373522 queued and waiting for resources
srun: job 4373522 has been allocated resources
Compiling ./futhark/nbody.fut...
Reporting arithmetic mean runtime of at least 10 runs for each dataset (min 0.5s).
More runs automatically performed for up to 300s to ensure accurate measurement.

./futhark/nbody.fut (no tuning file):
k=10, n=1000:          344μs (95% CI: [     343.3,      345.0])
k=10, n=10000:       18222μs (95% CI: [   18146.0,    18256.1])
k=10, n=100000:    1118006μs (95% CI: [ 1115029.3,  1121371.4])
```

Sometimes you have to wait for a little bit, but it seems like the
cluster is not heavily used.

One slight subtlety is that the user accounts do not have properly
configured environment variables for running CUDA. I found it
necessary to add the following to `$HOME/.profile`:

```
export LIBRARY_PATH=/usr/local/cuda/lib64
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CPATH=/usr/local/cuda/include
```

I also found it useful to add the following so that locally installed
programs were in PATH:

```
export PATH=$HOME/.local/bin:$PATH
```

## Futhark

### Setup

Grab an appropriate Futhark compiler tarball, e.g.

  https://futhark-lang.org/releases/futhark-nightly-linux-x86_64.tar.xz

unpack it, enter the directory, and run

  $ make PREFIX=$HOME/.local

and then the `futhark` command will work if `$HOME/.local/bin` is in
`$PATH`.

You should also install [ISPC](https://ispc.github.io/ispc.html) in
order for the `ispc` backend to work. Download

  https://github.com/ispc/ispc/releases/download/v1.23.0/ispc-v1.23.0-linux.tar.gz

and unpack it and copy the `bin/ispc` program to somewhere in your
`PATH`.

### Automatic benchmarking

[`bench_futhark.sh`](util/bench_futhark.sh).

### Manual benchmarking

Use `futhark bench prog.fut` where `prog.fut` is the Futhark program
in question. Pass `--backend=multicore` or `--backend=cuda` or
`--backend=ispc` as appropriate.

## SaC / OpenMP

I also have a `${HOME}/.local` directory for locally installed stuff. I compiled
`variance` from `util` and copy both this program and the `bench_...` scripts
to the path. 

Before benchmarking, it is important to compile the programs on cn132 via a 
batch script or `srun`.

Then it is a matter of `sbatch bench_omp.sh ...` or 
`sbatch bench_sac_mt.sh ...` to run the benchmarks. If the scripts are run 
without arguments, they print their usage.

## DaCe

### Setup

To install DaCe and run the benchmarks, Python is needed. Version 3.7 and above is currently supported, but newer
versions are recommended. A virtual environment is also optional, but recommended. We will provide later automated
scripts for creating a virtual environment and installing DaCe.

Get DaCe from GitHub and install it as follows:
```bash
git clone --recurse-submodules -b auto-opt-fixes https://github.com/spcl/dace.git
cd dace
python -m pip install -e .
```

In GPU benchmarks, CuPy is used to copy input data to GPU global memory. To install CuPy, run:
```bash
python -m pip install cupy-cuda12x
```

### Benchmarking

To run the benchmarks, execute from the top-level directory (`CFAL-bench`) the following script:
```bash
(sbatch) util/bench_dace.sh BENCHMARK_NAME
```
where `BENCHMARK_NAME` is one of the following:
- `FlashAttention`
- `LocVolCalib`
- `MG`
- `nbody-naive`
