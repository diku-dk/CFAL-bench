# Quickhull

Quickhull is a common geometric algorithm to compute the convex hull of a set of, in this case two-dimensional, points.
The sequential algorithm is for instance discussed on [Wikipedia](https://en.wikipedia.org/wiki/Quickhull).

As its name suggests, there is a similarity with quicksort. It is also a recursive algorithm, that performs two recursive calls over parts of the data.

The divide-and-conquer recursion can be flattened, to allow efficient execution on GPUs. This is for instance presented in [Fast Two Dimensional Convex Hull on the GPU](https://faculty.iiit.ac.in/~kkishore/ainaHull.pdf) (Pseudocode in algorithm 2).

This algorithm is an example of an irregular application, which at first may look unsuitable for array languages. With (possibly manual) flattening it is possible to implement this in array languages. This benchmark thus illustrates that array languages have a broader range of possible applications.

For simplicity, we can work with 2-dimensional points, but it should also be possible to extend this to three dimensions.

## Flattening
To adapt the divide-and-conquer algorithm to GPUs, we apply [NESL-style flattening](https://www.cs.cmu.edu/~guyb/papers/Nesl3.1.pdf). Instead of going in recursion on two segments, we let the GPU work on the entire array and mark segments into that array. As described in *Fast Two Dimensional Convex Hull on the GPU*, each element is annotated with a *label*, which is the index or identifier of a segment.

In an iteration of the outer loop, the algorithm performs folds and scans. It also filters/compacts/partitions the data, which can be implemented using a scan and permute/scatter.

These scans, which should happen per segment, can be lifted to segmented arrays using *operator lifting* (section 1.5 of [Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)).
Segmented folds can be implemented using segmented scans.

## Implementation details
In the Accelerate implementation we made the following assumptions:
- Indices and labels are represented as 64-bit integers.
- Coordinates are double precision floating point numbers.

Handling rounding errors requires that when we test if the distance of a point P to a line (A,B) is larger than zero, we must also check whether P != A and P != B.

## Test inputs
- 1M_rectangle_16384 has 1 million points, in a 16384x16384 rectangle. The convex hull has 16 points: (0.0,618.0),(0.0,15686.0),(1.0,16325.0),(17.0,16348.0), ...
- 1M_circle_16384 has 1 million points, placed on a 16384x16384 grid, sampled in a circle. The convex hull has 329 points: (1.0,8145.0),(3.0,8408.0),(14.0,8663.0),(28.0,8857.0), ...
- 1M_quadratic_2147483648 has 1 million points, in a 2147483648x2147483648 grid, roughly following a quadratic curve. The convex hull has 128069 points.
