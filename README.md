# CFAL-bench

This is the repository containing the benchmark work related to the upcoming collaborative paper "Comparing Functional Array Languages: Programming and GPU Performance"


## Benchmark Collection

In order to suggest a new benchmark, please do perform the following steps:

* choose an acronym
* create a subdirectory of that name
* add a Readme.md which contains:
   * a short description what it does / implements, highlighting the envisioned challenges
   * some rationale, why you think it is suitable for what we want to investigate
   * some rationale, why you believe that the wider community may accept this as a BM
   * pointers to related materials (papers / existing implementations)
* add a naive implementation which others can use to determine what is actually involved.
  This can be in any programming language but it should be small enough to allow for a quick
  assessment even if people are *not* familiar with the language used
* add the acronym into the table below and indicate what is available so far; the column "inclusion in CFAL"
  indicates whether there exists a consensus amongst us to include this BM. The language specific columns
  currently should only include whether an implementation in that language has been uploaded or not, it is not meant
  to indicate whether a super efficient one has been put there

## Benchmark Status

| BM name     | description | seq/par baselines | inclusion in CFAL   | Accelerate | APL | DaCe   | Futhark | SaC |
| ----------- | ----------- | ----------------- | ------------------- | ---------- | --- | ------ | ------- | --- |
| MG          | done        | GPU, Fortran, SAC | VERY LIKELY         | YES        | YES | YES    | YES     | YES |
| nbody-naive | done        | Seq C             | under consideration | YES        | YES | YES    | YES     | YES |
| Fin-LocVolC | done        | GPU,OMP, Haskell  | under consideration |            | YES | Almost | YES     | YES |
| FFT         | done        |                   |                     |            |     |        |         |     |
| Quickhull   | done        | qhull             |                     | YES        | YES |        | YES     | Multiple |
| FlashAttention | done     | Seq/MC C?         |                     |            | YES | YES    |         | YES |

