"""
    Naive python implementation of FlashAttention benchmark
    
    Note: this is in single precision
"""

import numpy as np
import math
import time
import argparse


def scale(x, m: int, n: int):
    for i in range(m):
        weight = np.sum(x[i, :])
        weight = 1 / weight
        x[i, :] *= weight


def stabilize(x, m: int, n: int):
    for i in range(m):
        maximum = np.max(x[i, :])
        x[i, :] -= maximum


def exp_arr(x):
    return np.exp(x)


def matmul(a, b):
    return a @ b


def matmulT(a, b):
    return a @ np.transpose(b)


def l2(x):
    return math.sqrt(np.sum(x * x))


def FlashAttention(Q, K, V, O, P_block, d, N):
    # Q, K, V, and O are Nxd matrices
    # P is a d x d matrix

    O_i = np.empty((d, d)).astype(np.float32)
    for i in range(N // d):
        #P_block = matmul(Q[i], K^t). Q[i] is d x d, K is N x d.
        # Note: original implementation uses pointers to select the block
        Q_i = Q[i * d:(i + 1) * d, :]
        P_block = matmulT(Q_i, K)

        stabilize(P_block, d, N)
        P_block = exp_arr(P_block)
        scale(P_block, d, N)

        # P_block is d x N, V is N x d.
        O_i = matmul(P_block, V)
        O[i * d:(i + 1) * d, :] = O_i


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


def standard_attention(Q, K, V):

    N, d = Q.shape

    S = Q @ K.T
    P = softmax(S)
    O = P @ V

    return O


def flash_attention(Q, K, V, M):

    N, d = Q.shape
    Bc = int(np.ceil(M / (4 * d)))
    Br = min(Bc, d)
    Tc = int(np.ceil(N / Bc))
    Tr = int(np.ceil(N / Br))

    O = np.zeros((N, d), dtype=Q.dtype)
    l = np.zeros((N,), dtype=Q.dtype)
    m = np.full((N,), -np.inf, dtype=Q.dtype)

    for j in range(Tc):

        Kj = K[j * Bc:min((j + 1) * Bc, N), :]
        Vj = V[j * Bc:min((j + 1) * Bc, N), :]

        for i in range(Tr):

            Qi = Q[i * Br:min((i + 1) * Br, N), :]
            Oi = O[i * Br:min((i + 1) * Br, N), :]
            li = l[i * Br:min((i + 1) * Br, N)]
            mi = m[i * Br:min((i + 1) * Br, N)]

            Sij = Qi @ Kj.T
            mij = np.max(Sij, axis=1)
            Pij = np.exp(Sij - mij.reshape(len(mij), 1)) 
            lij = np.sum(Pij, axis=1)

            mi_new = np.maximum(mi, mij)
            li_new = np.exp(mi - mi_new) * li + np.exp(mij - mi_new) * lij

            # Oi[:] = np.diag(1 / li_new) @ (np.diag(li) @ np.exp(mi - mi_new).reshape((len(mi), 1)) * Oi + np.exp(mij - mi_new).reshape((len(mij), 1)) * Pij @ Vj)
            Oi[:] = (
                    (li.reshape(len(li), 1) * np.exp(mi - mi_new).reshape((len(mi), 1))) * Oi +
                    np.exp(mij - mi_new).reshape((len(mij), 1)) * Pij @ Vj
                ) / li_new.reshape(len(li_new), 1)
            li[:] = li_new
            mi[:] = mi_new

    return O


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('d', type=int, help="d")
    parser.add_argument('N', type=int, help="N")
    args = vars(parser.parse_args())
    N = args["N"]
    d = args["d"]

    assert N % d == 0, "N must divide d"

    # Create data. The input sequences (Q, K, and V) and output one (O) are Nxd matrices,
    # P_block is d x N
    rng = np.random.default_rng(42)
    # Q = np.ones((N, d)).astype(np.float32)
    # K = np.ones((N, d)).astype(np.float32)
    # V = np.ones((N, d)).astype(np.float32)
    Q = rng.random((N, d)).astype(np.float32)
    K = rng.random((N, d)).astype(np.float32)
    V = rng.random((N, d)).astype(np.float32)

    O_ref = np.empty((N, d)).astype(np.float32)
    O = np.empty((N, d)).astype(np.float32)
    P_block = np.empty((d, N)).astype(np.float32)

    ### Start
    start = time.time()
    O_ref = standard_attention(Q, K, V)
    end = time.time()

    ### Check

    # assert np.allclose(np.linalg.norm(O), math.sqrt(N * d)), f"L2 norm is {l2(O)}, it should be {math.sqrt(N*d)}"

    ### Perf
    print(f"Time in usecs: {(end-start)*1e6}")
    duration = end - start
    ops = 2.0 * N * N * (d + 1)
    print(f"Compute rate: {ops/duration/1e9} Gflops/s\n")

    ### Start
    start = time.time()
    FlashAttention(Q, K, V, O, P_block, d, N)
    end = time.time()

    ### Check

    # assert np.allclose(np.linalg.norm(O), math.sqrt(N * d)), f"L2 norm is {l2(O)}, it should be {math.sqrt(N*d)}"
    # assert np.allclose(O, O_ref)
    print(np.linalg.norm(O - O_ref) / np.linalg.norm(O_ref))

    ### Perf
    print(f"Time in usecs: {(end-start)*1e6}")
    duration = end - start
    ops = 2.0 * N * N * (d + 1)
    print(f"Compute rate: {ops/duration/1e9} Gflops/s\n")

    M0 = 4 * N * d
    for M in (M0, M0 // 2, M0 // 4, M0 // 8, M0 // 16, M0 // 32):
        print(f"M = {M}")

        ### Start
        start = time.time()
        O = flash_attention(Q, K, V, M)
        end = time.time()

        ### Check

        # assert np.allclose(np.linalg.norm(O), math.sqrt(N * d)), f"L2 norm is {l2(O)}, it should be {math.sqrt(N*d)}"
        # assert np.allclose(O, O_ref)
        print(np.linalg.norm(O - O_ref) / np.linalg.norm(O_ref))

        ### Perf
        print(f"Time in usecs: {(end-start)*1e6}")
        duration = end - start
        ops = 2.0 * N * N * (d + 1)
        print(f"Compute rate: {ops/duration/1e9} Gflops/s\n")
