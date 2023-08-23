# Delta-Stepping

Δ-stepping computes the single-source shortest-path (SSSP) problem in a
[directed] graph. It is sort of a parallel version of Dijkstra's algorithm, and
one of the three Graph500 benchmarks.

[Wikipedia](https://en.wikipedia.org/wiki/Parallel_single-source_shortest_path_algorithm)

## Pros
- Represents an import class of problems
- Might be a bit difficult to implement in Accelerate due to it being an
  embedded language, so we get something to talk about in the language
  description part. I don't think this will be a problem for anybody else
  though.

## Cons
- More of a computational kernel than an application, but still relevant since
  it's used in Graph500

