# 3D FFT

## Algorithm description

The input and output are 3D arrays of complex numbers. The space of algorithms is very large,
so I will just describe one commonly used. We first compute the 1D FFT on X[-, j, k] for all
(j, k), then the 1D FFT on X[i, -, k] for all (i, k) and finally on X[i, j, -] for all (i, j).
The 1D FFT is a recursive

## Relevancy

3D FFT is part of [NAS parallel benchmarks](https://www.nas.nasa.gov/software/npb.html)
and [Berkely Dwarfs](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-183.pdf).
The latter has been cited over 3000 times so it is widely accepted in the
community. FFT also made it to the top 10 algorithms of the 20th century list.

## Performance characteristics

The algorithm is O(n log n) in the total amount of elements, and the locality of reference is
bad, making this a bandwidth bound problem. Recursive and iterative algorithms for the 1D FFT
exist. Key to getting performance is to make proper use of the memory hierarchy.

## Implementation examples

TODO
