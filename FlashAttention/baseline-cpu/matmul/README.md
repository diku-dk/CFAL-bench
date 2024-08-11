# Motivation

We perform O(N^2 d) flops in matrix multiplications, and only O(Nd) in the
softmax, so optimising matmul is vital.

BLAS will
    i)   check the arguments for consistency
    ii)  choose a code-path depening on alpha, beta, etc.
    iii) pack a, b into buffers and allocating them

i) and ii) are obviously not necessary, and can give considerable overhead
because we do so many small matmuls.
As the matmuls happen on small contiguous memory, TLB misses should not be
a problem, so iii) is also unnecessary.

Implementing our own version will avoid these overheads.
