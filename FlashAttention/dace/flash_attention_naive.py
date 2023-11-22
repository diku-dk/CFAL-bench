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
        weight = np.float32(0.0)
        for j in range(n):
            weight += x[i, j]  

        weight = 1 / weight
        for j in range(n):
            x[i, j] *= weight


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
    P_i = np.empty((d, d)).astype(np.float32)
    O_i = np.empty((d, d)).astype(np.float32)
    for i in range(N // d):
        #P_block = matmul(Q[i], K^t). Q[i] is d x d, K is N x d.
        # TODO: original implementation uses pointers to select the block
        Q_i = Q[i * d:(i + 1) * d, :]
        P_block = matmulT(Q_i, K)
        
        stabilize(P_block, d, N)
        P_block = exp_arr(P_block)
        scale(P_block,d,N)

        # P_block is d x N, V is N x d.
        O_i = matmul(P_block, V)
        O[i * d:(i + 1) * d, :] = O_i


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
    Q = np.ones((N, d)).astype(np.float32)
    K = np.ones((N, d)).astype(np.float32)
    V = np.ones((N, d)).astype(np.float32)

    O = np.empty((N, d)).astype(np.float32)
    P_block = np.empty((d, N)).astype(np.float32)

    ### Start
    start = time.time()
    FlashAttention(Q, K, V, O, P_block, d, N)
    end = time.time()
    
    ### Check

    assert np.allclose(np.linalg.norm(O), math.sqrt(N*d)) , f"L2 norm is {l2(O)}, it should be {math.sqrt(N*d)}"

    ### Perf
    print(f"Time in usecs: {(end-start)*1e6}")
    duration = end-start
    ops = 2.0 * N * N * (d + 1)
    print(f"Compute rate: {ops/duration/1e9} Gflops/s\n")
