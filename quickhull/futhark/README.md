# Futhark implementation

Flattened and fairly efficient.

First run `make data` to produce Futhark-readable datafiles via the
[`input2fut.py`](input2fut.py) program.

Then run e.g. `futhark bench --backend=cuda quickhull.fut` to
benchmark.
