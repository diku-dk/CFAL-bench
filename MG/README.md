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

## Relevancy

Multigrid is part of [NAS parallel benchmarks](https://www.nas.nasa.gov/software/npb.html)
and [Berkely Dwarfs](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-183.pdf).
The latter has been cited over 3000 times so it is widely accepted in the
community.

## Performance characteristics

The main datastructure used is a 3D array, and most of the execution time is
in a 27-point relaxation step as those in iterative stencils. In SaC this
looks roughly like

```
double[.,.,.] relax(input[.,.,.], weights[3, 3, 3], int k)
{
    /* Real function has periodic boundary conditions */
    return {iv -> sum({jv -> input[iv - [1, 1, 1] + k * jv)] | jv < [3, 3, 3]}
                      * weights)};
}
```

We have a reasonable number of flops per element, but the locality of reference
is bad. There is not really much we can do to improve temporal locality, so
tiling is not necessary to match imperative solutions.

## Implementation examples

Reference implementation from NPB:
[Fortran + OMP](https://github.com/casys-kaist/NPB3.4/blob/master/NPB3.4-OMP/MG/mg.f)

Conciser and annotated implementation
[Chapel](https://github.com/chapel-lang/chapel/blob/main/test/npb/mg/mg-annotated.chpl)

8 year old SaC
[SaC](https://github.com/SacBase/NASParallelBenchmarks/blob/master/MG/mg_rotate.sac)

## TODO

Provide a simple reference implementation in a more well-known language.
