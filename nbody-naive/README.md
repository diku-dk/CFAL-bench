# Naive n-body

**This is not a serious suggestion for a benchmark (it is likely too
trivial), but a strawman that the language comparison team can use if
it takes too long to decide the real benchmarks.**

The problem is simulating *k* sequential time steps of duration *dt*,
of *n* point particles in 3-dimensional space, using the naive *O(n²)*
algorithm.

A particle consists of its position, its velocity, and its mass.  A
particle is initially assumed to have zero velocity.

The input should be the numbers *k* and *dt* and the initial particle
positions and masses.

The output should be the final particle positions.

The reported time should include only the simulation, i.e. not reading
the input, writing the output, or doing other initialisation.

All floating-point arithmetic should be in double precision.
