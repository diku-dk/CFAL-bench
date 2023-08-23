# Barnes-Hut simulation

An (approximate) n-body simulation that uses an octree data structure to perform
the calculation in O(n log n) time, compared to the O(n^2) direct sum
implementation (c.f. nbody-naive).

[Wikipedia](https://en.wikipedia.org/wiki/Barnes–Hut_simulation)

## Pros
- This is an application benchmark with obvious relevance
- Exposes weaknesses in (flat) array languages

## Cons
- Difficult to implement for flat array languages (also a pro)

