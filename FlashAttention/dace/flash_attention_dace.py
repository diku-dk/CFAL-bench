"""
    DaCe python implementation of FlashAttention benchmark
    
    Note: this is in single precision
"""

# TODOs:
# - MatmulT: avoid transposition

import numpy as np
import math
import time
import argparse
import dace
from dace.transformation.auto.auto_optimize import auto_optimize

N, d = (dace.symbol(s, dtype=dace.int32) for s in ('N', 'd'))


def scale(x, m: int, n: int):
    for i in range(m):
        weight = np.sum(x[i, :])
        weight = 1 / weight
        x[i, :] *= weight


def stabilize(x, m: int, n: int):
    for i in range(m):
        maximum = np.max(x[i, :])
        x[i, :] -= maximum


@dace.program
def exp_arr(x):
    return np.exp(x)


@dace.program
def matmul(a, b):
    return a @ b


@dace.program
def matmulT(a: dace.float32[d, d], b: dace.float32[N, d]):
    return a @ np.transpose(b)


def l2(x):
    return math.sqrt(np.sum(x * x))


@dace.program
def FlashAttention(Q: dace.float32[N, d], K: dace.float32[N, d], V: dace.float32[N, d], O: dace.float32[N, d]):
    # Q, K, V, and O are Nxd matrices
    # P is a d x d matrix

    O_i = np.empty((d, d), dtype=Q.dtype)
    P_block = np.empty((d, N), dtype=Q.dtype)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('d', type=int, help="d")
    parser.add_argument('N', type=int, help="N")
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help="Target platform")

    args = vars(parser.parse_args())
    N = args["N"]
    d = args["d"]
    target = args["target"]

    assert N % d == 0, "N must divide d"

    # Create data. The input sequences (Q, K, and V) and output one (O) are Nxd matrices,
    # P_block is d x N
    Q = np.ones((N, d)).astype(np.float32)
    K = np.ones((N, d)).astype(np.float32)
    V = np.ones((N, d)).astype(np.float32)

    O = np.empty((N, d)).astype(np.float32)
    P_block = np.empty((d, N)).astype(np.float32)

    # Parsing SDFG
    sdfg = FlashAttention.to_sdfg()
    if target == "cpu":
        sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.CPU)
    else:
        sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    sdfg.compile()

    ### Start
    start = time.time()
    sdfg(Q=Q, K=K, V=V, O=O, d=d, N=N)
    end = time.time()

    ### Check

    assert np.allclose(np.linalg.norm(O), math.sqrt(N * d)), f"L2 norm is {l2(O)}, it should be {math.sqrt(N*d)}"

    ### Perf
    print(f"Time in usecs: {(end-start)*1e6}")
    duration = end - start
    ops = 2.0 * N * N * (d + 1)
    print(f"Compute rate: {ops/duration/1e9} Gflops/s\n")
