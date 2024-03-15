# C baseline for FlashAttention

There are a few implementations of the attention computation here, as well as
some helper tools. Run `make` to build them all.

The attention implementations can either run stand-alone (by generating
matrices of ones and then computing with those), or alternatively read `Q`, `K`
and `V` from stdin and write `O` to stdout. The former mode is useful for
benchmarking; the latter mode is useful for correctness testing. Use the latter
mode by passing `-io` instead of `d` and `N` to an attention implementation.

- `generate`: Generate three matrices for the various attention
  implementations.
  Run `bin/generate` for the exact invocation instructions.
  The format is:
  - One line with two integers: `d` and `N`.
  - Three times (for Q, K and V):
    - `N` lines of `d` floating-point numbers.
- `compare`: Compare two output matrices.
  Run `bin/compare` for the exact invocation instructions.
  The input format is simply `N` lines of `d` floating-point numbers.
- `flash`: Original baseline implementation. This algorithm is halfway between
  the paper's "Algorithm 0" (standard attention) and "Algorithm 1"
  (FlashAttention).
- `flash_mt`: Multithreaded version of `flash` using OpenMP.
- `flash_alg1`: Implementation of Algorithm 1, the actual Flash Attention
  algorithm, by Aaron.
- `attention`: Straightforward implementation of "Algorithm 0" (standard
  attention). Turns out to be faster than `flash` (on CPU).

Example session:
```sh
make
bin/generate 64 1024 >input.txt
bin/attention -io <input.txt >out-att.txt
bin/flash -io <input.txt >out-flash.txt
bin/compare 64 1024 out-att.txt out-flash.txt
```

## Dependencies

On Ubuntu 22.04, install `libblis64-openmp-dev` and ignore the rest of this
section.

On other Linux-likes, locally compile BLIS with the following commands:
```sh
# call the current directory <PATH_TO_BLIS>
git clone https://github.com/flame/blis
cd blis
./configure --prefix=$PWD/prefix --enable-cblas --enable-threading=openmp auto
```
Add the following lines to `Makefile`:
```
FLAGS += -I<PATH_TO_BLIS>/blis/prefix/include
LFLAGS += -L<PATH_TO_BLIS>/blis/prefix/lib
LFLAGS += -Wl,-rpath=<PATH_TO_BLIS>/blis/prefix/lib
```
replacing `<PATH_TO_BLIS>` with the path to the blis clone.

Then change the `-lblis64` to `-lblis` on the `LFLAGS =` line of `Makefile`.
