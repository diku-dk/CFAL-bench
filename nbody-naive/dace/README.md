# DaCe implementation

This folder contains:
- `nbody_dace.py` the DaCe (and numpy) implementation of the benchmark
- `nbody_naive.py` a naive python implementation (1:1 from sequential C version). This also dumps input data to a file to check correctness against the C version.

## How to run

To run the DaCe version:
```bash
python <num_iterations> <num_particles> <dt> --target cpu/gpu
```
The particles are randomly generated. By default it runs the benchmark for cpu.
Results can be validated against the numpy version appending the `-v` flag (this may take some time).

## TODOs
- optimize
- measure perf properly (once we all agree)