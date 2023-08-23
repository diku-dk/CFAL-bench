# Conjugate Gradient

See: https://www.nas.nasa.gov/assets/nas/pdf/techreports/1994/rnr-94-007.pdf

## Summary

This benchmark uses the inverse power method to find an estimate of
the largest eigenvalue of a symmetric positive definite sparse matrix with a
random pattern of nonzeros

## Relevance

Conjugate Gradient is part of [NAS parallel benchmarks](https://www.nas.nasa.gov/software/npb.html)
and [Berkely Dwarfs](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-183.pdf).
The latter has been cited over 3000 times so it is widely accepted in the
community.


## Implementation examples

Reference implementation from NPB:
[Fortran + OMP](https://github.com/casys-kaist/NPB3.4/blob/master/NPB3.4-OMP/CG/cg.f)

