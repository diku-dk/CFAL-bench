# Reference implementation of Quickhull

This program is derived from the Quickhull implementation from the
[Problem Based Benchmark
Suite](https://github.com/cmuparlay/pbbsbench). It has been lightly
modified to work outside of the PBBS framework, including reading our
data files and using different timing.

It makes use of the Parlay library, which is included as a Git
submodule. Do `git submodule init` and `git submodule update` to
download it.
