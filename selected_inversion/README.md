# Selected Inversion

Contains algorithms that compute only selected elements of the inverse of a matrix.
- Block tridiagonal solver: Given a sparse block tridiagonal matrix, computes the same nnz elements of the inverse.

## Characteristics

- A fast implementation (probably) requires BLAS/LAPACK calls.
- Tests the capability of the compiler/backend to schedule concurrently independent operations.

