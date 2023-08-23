# Multigrid Method

## Algorithm description

The Multigrid method is used to obtain an approximate solution to a discrete
Poisson problem. The benchmark consists of four iterations, each of which
evaluates a residual and applies a correction:

```
r = v - Au        (residual)
u = u + M^k r     (correction)
```

Here M^k is a recursive function dependent on k. For full spec see the
[tech report](https://www.nas.nasa.gov/assets/nas/pdf/techreports/1994/rnr-94-007.pdf).
The most important part performance-wise is explained in
```Performance characterstics```.

## Relevance

Multigrid is part of [NAS parallel benchmarks](https://www.nas.nasa.gov/software/npb.html)
and [Berkely Dwarfs](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-183.pdf).
The latter has been cited over 3000 times so it is widely accepted in the
community.

## Performance characteristics

The main datastructure used is a 3D array, and most of the execution time is
spent in a 27-point relaxation step with cyclic boundary conditions. In SaC, this
can be expressed as

```
double[n,n,n] relax(double[n,n,n] input, weights[3, 3, 3])
{
    return { iv -> sum ({jv -> weights[jv] * rotate (1-jv, input)[iv]})
                 | iv < [n,n,n] };
}
```
The relaxation steps are interspersed with coarsening and refinement steps
allowing for different approaches towards managing the memory involved.
This opens up many avenues for handling the memory eg whether to perform 
the coarser grids in place of the finer ones or whether to copy them out.
How the different languages handle this, how this can or cannot be influenced by the
programmer might be interesting here!

We have a reasonable number of flops per element, but the locality of reference
is relatively poor. There is not really much we can do to improve temporal locality, so
tiling is not necessary to match imperative solutions.

## Implementation examples

Reference implementation from NPB:
[Fortran + OMP](https://github.com/casys-kaist/NPB3.4/blob/master/NPB3.4-OMP/MG/mg.f)

Conciser and annotated implementation
[Chapel](https://github.com/chapel-lang/chapel/blob/main/test/npb/mg/mg-annotated.chpl)

20 year old SaC
[SaC](https://github.com/SacBase/NASParallelBenchmarks/blob/master/MG/mg_rotate.sac)

## TODO

Provide a simple yet complete reference implementation.
