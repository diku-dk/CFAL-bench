# Motivation

We perform O(N^2 d) flops in matrix multiplications, and only O(Nd) in the
softmax, so optimising matmul is vital. As we do many small matmuls, the
error checking and allocating of blas routines hurts performance. Commenting
out everything except blas calls shows that we obtain about 40% peak 
(with OpenBLAS) this way. So let us implement this ourselves so we can 
reuse buffers, avoid error checking, and make some assumptions about the input.
