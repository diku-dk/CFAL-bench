""" Vanilla attention GNN model single layer forward pass."""
import numpy as np
from scipy import sparse


def forward_naive(A: sparse.csr_matrix,  # Graph adjacency matrix
                  H: np.ndarray,  # Input features
                  W: np.ndarray  # Weight matrix
                 ) -> np.ndarray:  # Output features
    
    """
    H' = ReLU((A * (H x H.T)) x H x W)
    `*` is Hadamard product
    `x` is matrix multiplication
    """

    HHT = H @ H.T
    S = A.toarray() * HHT
    SHW = S @ (H @ W)
    H_prime = np.maximum(SHW, 0)

    return H_prime


def forward_SDDMM(A: sparse.csr_matrix,  # Graph adjacency matrix
                  H: np.ndarray,  # Input features
                  W: np.ndarray  # Weight matrix
                 ) -> np.ndarray:  # Output features
    
    """
    H' = ReLU((A * (H x H.T)) x H x W)
    `*` is Hadamard product
    `x` is matrix multiplication
    """

    S_data = np.empty_like(A.data, dtype=H.dtype)
    for i in range(A.indptr.size - 1):
        start = A.indptr[i]
        finish = A.indptr[i+1]
        for j in range(start, finish):
            S_data[j] = H[i] @ H[A.indices[j]]
    S = sparse.csr_matrix((S_data, A.indices, A.indptr), shape=A.shape, dtype=H.dtype)

    SHW = S @ (H @ W)
    H_prime = np.maximum(SHW, 0)

    return H_prime


if __name__ == "__main__":

    rng = np.random.default_rng(42)
    N = 16
    d = 4
    A = sparse.random(N, N, density=0.1, format='csr', dtype=np.float32, random_state=rng)
    A.data[:] = 1
    H = rng.random((N, d), dtype=np.float32)
    W = rng.random((d, d), dtype=np.float32)

    H_prime_naive = forward_naive(A, H, W)
    H_prime_SDDMM = forward_SDDMM(A, H, W)

    assert np.allclose(H_prime_naive, H_prime_SDDMM)
