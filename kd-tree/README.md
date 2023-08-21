## K-d Tree Construction

A Haskell high-level specification in the nested parallel style is
included in folder Haskell. A GPU implementation needs to flatten
the irregular parallelism.

If this is accepted as a benchmark Cosmin volunteers to quickly
add a Cuda and a Futhark (prototype) implementation, that documents
various implementation strategies and tradeoffs. 

```
Input
  (height : int)         the height of the k-d tree
  (refs: [n][d]float)    n d-dimensional reference points

Result
  (med_dims : [q]int)    the indices and
  (med_vals : [q]float)  the median value of the split dimension
                         corresponding to each internal node
                         in the k-d tree (q = 2^{height} - 1)
  (leaves : [n][d]float) the flat-array containing the reordered
                         reference points.   Semantically `leaves`
                         is an array containing 2^{height} subarrays
                         of variant lenghts, but such that the subarray
                         lengths sum up to `n`.
  (shape  : [q+1]int)    the shape of the `leaves` array of subarrays,
                           i.e., the number of elements in each of the
                           `2^{height} == q+1` subarrays.
                         alternatively, it can be the start offset of
                         each sub-array (similar to CSR representation).
Algorithm:

for each breadth level i = 0 .. height - 1 do sequentially:
  for all tree nodes on level i (there are 2^i of them) do in parallel:
    1. find `med_dim`, the dimension of widest spread (for the node's points)
    2. find `med_val`, the median value of dimension `med_dim` (for the node's points)
    3. partition the points of the current node into two subarrays:
         -- the first  containing the points whose value on `med_dim` is less than `med_val`
         -- the second containing the points whose value on `med_dim` is greater or equal to `med_val`
       this creates the next level in the kd tree
       of course the value of `med_dim` and `med_val` for each internal node are saved
          (in `med_dims` and `med_vals`).
``` 

Seminal article:

[1] J. L Bentley, "Multidimensional Binary Search Trees Used For Associative Searching", Communications of the ACM, vol 18(9), 1975.

Exact and approximate k-nearest neighbors by means of kd-tree propagation have been applied in fields such as astro physics and image processing, for example see:

[2] "Buffer k-d trees: processing massive nearest neighbor queries on GPUs", ICML'14

[3] "Computing nearest-neighbor fields via propagation assisted kd-trees", Computer Vision and Pattern Recognition, 2012

[4] "Randomized approximate nearest neighbors algorithm", PNAS 2011.
