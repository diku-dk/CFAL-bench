## Rodinia's LUD implementation

Linear-algebra kernel emphasizing the tiling optimization. It also exhibits some segmented-reduce parallelism.
The rationale would be to evaluate what performance can functional/data-flow languages offer on polyhedral-like code.

Of note:

- the block size can be freely chosen (named BLOCK in main.cu), as long as the matrix dimension is a multiple of it. 

- to my knowledge, the Cuda implementation is not derivable by means of compiler transformations from the golden-sequential implementation, i.e., it is an algorithmic change (blocking).  Golden sequential is currently only useful for validating the result.

- ToDo: will write a golden sequential that resembles the Cuda implementation.

```
Input 
  (b : Int)           denotes the block size
  (ass : [n][n]float) the matrix; `n` is a multiple of `b`
Result
  (bss : [n][n]float) the LU decomposition of ass.
```
