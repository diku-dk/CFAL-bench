# LULESH

LULESH (Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics) is a
mini-application introduced by LLNL as a proxy for investigating & demonstrating
the performance of different programming languages/models/machines on real
[magneto-]hydrodynamic simulation codes.

The [website](https://asc.llnl.gov/codes/proxy-apps/lulesh) contains many
reference implementations, as well as documentation on motivation, algorithm,
etc. There is also a
[paper](https://cs.stanford.edu/~zdevito/Programming_Model_Comparison.pdf) that
compares LULESH implemented in several emerging programming languages that is of
interest for us.

## Pros
- This is an _application benchmark_ with obvious relevance
- Plenty of opportunity for high-level programming languages to beat the C/C++
  implementation (c.f. above paper)

## Cons
- This is not a trivial one-liner to implement like NAS-MG

