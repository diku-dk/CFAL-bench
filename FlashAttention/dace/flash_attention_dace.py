import argparse
import dace
import numpy as np
import os
import time

from dace.transformation.auto.auto_optimize import auto_optimize


import warnings
warnings.filterwarnings("ignore")

def stable_softmax_vector(x):
    m = np.max(x)
    d = np.sum(np.exp(x - m))
    return np.exp(x - m) / d


def stable_softmax_matrix(x):
    y = np.empty_like(x)
    for i in range(x.shape[0]):
        y[i] = stable_softmax_vector(x[i])
    return y


def stable_softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def online_softmax_vector(x):
    m = -np.inf
    d = 0
    for i in range(x.shape[0]):
        m_old = m
        d_old = d
        m = max(m, x[i])
        d = d_old * np.exp(m_old - m) + np.exp(x[i] - m)
    return np.exp(x - m) / d


def online_softmax_matrix(x):
    y = np.empty_like(x)
    for i in range(x.shape[0]):
        y[i] = online_softmax_vector(x[i])
    return y


def online_softmax(x):
    y = np.empty_like(x)
    m = np.full(x.shape[0], -np.inf, x.dtype)
    d = np.zeros(x.shape[0], x.dtype)
    m_old = np.empty(x.shape[0], x.dtype)
    d_old = np.empty(x.shape[0], x.dtype)
    for j in range(x.shape[1]):
        m_old[:] = m
        d_old[:] = d
        m = np.maximum(m_old, x[:, j])
        d = d_old * np.exp(m_old - m) + np.exp(x[:, j] - m)
    y[:] = np.exp(x - m[:, np.newaxis]) / d[:, np.newaxis]
    return y


def standard_attention(Q, K, V):
    S = Q @ K.T
    P = stable_softmax(S)
    return P @ V


N, d = (dace.symbol(s) for s in ('N', 'd'))
Ti, Tj = (dace.symbol(s) for s in ('Ti', 'Tj'))


@dace.program
def stable_softmax_dace(x: dace.float32[N, N]):
    e_x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
    return e_x / np.sum(e_x, axis=1)[:, np.newaxis]


@dace.program
def standard_attention_dace(Q: dace.float32[N, d], K: dace.float32[N, d], V: dace.float32[N, d]):
    S = Q @ K.T
    P = stable_softmax_dace(S)
    return P @ V


def standard_attention_online(Q, K, V):
    S = Q @ K.T
    P = online_softmax(S)
    return P @ V


def flash_attention(Q, K, V):

    S = Q @ K.T

    m = np.full(S.shape[0], -np.inf, S.dtype)
    d = np.zeros(S.shape[0], S.dtype)
    O = np.zeros_like(Q)
    m_old = np.empty(S.shape[0], S.dtype)
    d_old = np.empty(S.shape[0], S.dtype)
    for j in range(S.shape[1]):
        m_old[:] = m
        d_old[:] = d
        m[:] = np.maximum(m_old, S[:, j])
        d[:] = d_old * np.exp(m_old - m) + np.exp(S[:, j] - m)
        O[:] = (O * d_old[:, np.newaxis] * np.exp(m_old - m)[:, np.newaxis] + np.exp(S[:, j] - m)[:, np.newaxis] @ V[j, :][np.newaxis, :]) / d[:, np.newaxis]

    return O


@dace.program
def flash_attention_dace(Q: dace.float32[N, d], K: dace.float32[N, d], V: dace.float32[N, d]):

    S = Q @ K.T

    m = np.full([S.shape[0]], -np.inf, S.dtype)
    l = np.zeros([S.shape[0]], S.dtype)
    O = np.zeros_like(Q)

    for j in range(S.shape[1]):
        for i in dace.map[0:S.shape[0]]:
            m_old = m[i]
            l_old = l[i]
            m[i] = max(m_old, S[i, j])
            l[i] = l_old * np.exp(m_old - m[i]) + np.exp(S[i, j] - m[i])
            O[i, :] = (O[i, :] * l_old * np.exp(m_old - m[i]) + np.exp(S[i, j] - m[i]) * V[j, :]) / l[i]
    
    return O


@dace.program
def flash_attention_dace_2(Q: dace.float32[N, d], K: dace.float32[N, d], V: dace.float32[N, d]):

    m = np.full([N], -np.inf, Q.dtype)
    l = np.zeros([N], Q.dtype)
    O = np.zeros_like(Q)

    for j in range(N):
        for i in dace.map[0:N]:
            m_old = m[i]
            l_old = l[i]
            Sij = Q[i, :] @ K[j, :]
            m[i] = max(m_old, Sij)
            l[i] = l_old * np.exp(m_old - m[i]) + np.exp(Sij - m[i])
            O[i, :] = (O[i, :] * l_old * np.exp(m_old - m[i]) + np.exp(Sij - m[i]) * V[j, :]) / l[i]
    
    return O


@dace.program
def flash_attention_dace_3(Q: dace.float32[N, d], K: dace.float32[N, d], V: dace.float32[N, d]):

    m = np.full([N], -np.inf, Q.dtype)
    l = np.zeros([N], Q.dtype)
    O = np.zeros_like(Q)

    for ti in dace.map[0:N:Ti]:
        for tj in range(0, N, Tj):
            S = Q[ti:ti+Ti, :] @ np.transpose(K[tj:tj+Tj, :])
            for j in range(tj, tj+Tj):
                m_old = np.copy(m[ti:ti+Ti])
                l_old = np.copy(l[ti:ti+Ti])
                Sij = S[:, j - tj]
                m[ti:ti+Ti] = np.maximum(m_old, Sij)
                exp0 = l_old * np.exp(m_old - m[ti:ti+Ti])
                exp1 = np.exp(Sij - m[ti:ti+Ti])
                exp1V = np.multiply.outer(exp1, V[j, :])
                m[ti:ti+Ti] = np.maximum(m_old, Sij)
                l[ti:ti+Ti] = exp0 + exp1
                O[ti:ti+Ti, :] = (O[ti:ti+Ti, :] * exp0[:, np.newaxis] + exp1V) / np.reshape(l[ti:ti+Ti], (Ti, 1))
    
    return O


@dace.program
def flash_attention_dace_4(Q: dace.float32[N, d], K: dace.float32[N, d], V: dace.float32[N, d], O: dace.float32[N, d]):

    # m = np.full([N], -np.inf, Q.dtype)
    # l = np.zeros([N], Q.dtype)
    # O = np.zeros_like(Q)

    for ti in dace.map[0:N:Ti]:

        m = np.full([Ti], -np.inf, Q.dtype)
        l = np.zeros([Ti], Q.dtype)
        S = np.empty([Ti, Tj], Q.dtype)
        Oi = np.zeros([Ti, d], Q.dtype)

        Qi = Q[ti:ti+Ti, :]

        for tj in range(0, N, Tj):

            S[:] = Qi @ np.transpose(K[tj:tj+Tj, :])

            max_row = np.max(S, axis=1)
            m_new = np.maximum(m, max_row)
            p_tilde = np.exp(S - m_new[:, np.newaxis])
            sum_row = np.sum(p_tilde, axis=1)
            l_tmp = l * np.exp(m - m_new)
            l_new = l_tmp + sum_row
            Oi[:] = (Oi * l_tmp[:, np.newaxis] + p_tilde @ V[tj:tj+Tj, :]) / l_new[:, np.newaxis]
            m[:] = m_new
            l[:] = l_new
        
        O[ti:ti+Ti, :] = Oi



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="(Flash)Attention")
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-d", type=int, default=64)
    parser.add_argument("-Ti", type=int, default=64) 
    parser.add_argument("-Tj", type=int, default=64)
    args = parser.parse_args()

    N = args.N
    d = args.d
    assert N % d == 0, "N must divide d"

    validate = True

    rng = np.random.default_rng(42)

    Q = rng.random((N, d), dtype=np.float32)
    K = rng.random((N, d), dtype=np.float32)
    V = rng.random((N, d), dtype=np.float32)

    # try:
    #     import cupy as cp
    #     Q_dev = cp.asarray(Q)
    #     K_dev = cp.asarray(K)
    #     V_dev = cp.asarray(V)
    #     test_gpu = True
    # except (ModuleNotFoundError, ImportError):
    #     test_gpu = False
    test_gpu = False

    if validate:
        O_std_ref = standard_attention(Q, K, V)

    sdfg = standard_attention_dace.to_sdfg(simplify=False)
    sdfg.simplify()
    sdfg.expand_library_nodes()
    auto_optimize(sdfg, dace.DeviceType.CPU)
    func = sdfg.compile()
    O_dace = func(Q=Q, K=K, V=V, N=N, d=d)
    if validate:
        print(np.linalg.norm(O_std_ref - O_dace) / np.linalg.norm(O_std_ref))
        assert np.allclose(O_std_ref, O_dace)
    start = time.perf_counter()
    for i in range(10):
        O_dace = func(Q=Q, K=K, V=V, N=N, d=d)
    finish = time.perf_counter()
    print("Standard attention DaCe mean execution: ", (finish - start) / 10, "seconds", flush=True)
    gflops = 4 * N * N * (d + 5) / ((finish - start)/10) / 1e9
    print("Standard attention DaCe mean GFLOP/s: ", gflops, flush=True)
    print()

    # sdfg = flash_attention_dace.to_sdfg(simplify=False)
    # sdfg.simplify()
    # auto_optimize(sdfg, dace.DeviceType.CPU)
    # func = sdfg.compile()
    # O_dace = func(Q=Q, K=K, V=V, N=N, d=d)
    # if validate:
    #     print(np.linalg.norm(O_std_ref - O_dace) / np.linalg.norm(O_std_ref))
    #     assert np.allclose(O_std_ref, O_dace)
    # start = time.perf_counter()
    # for i in range(10):
    #     O_dace = func(Q=Q, K=K, V=V, N=N, d=d)
    # finish = time.perf_counter()
    # print("Flash attention DaCe mean execution: ", (finish - start) / 10, "seconds", flush=True)
    # gflops = 4 * N * N * d / ((finish - start)/10) / 1e9
    # print("Flash attention DaCe mean GFLOP/s: ", gflops, flush=True)
    # print()

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # sdfg = flash_attention_dace_2.to_sdfg(simplify=False)
    # sdfg.simplify()
    # auto_optimize(sdfg, dace.DeviceType.CPU)
    # func = sdfg.compile()
    # O_dace = func(Q=Q, K=K, V=V, N=N, d=d)
    # if validate:
    #     print(np.linalg.norm(O_std_ref - O_dace) / np.linalg.norm(O_std_ref))
    #     assert np.allclose(O_std_ref, O_dace)
    # start = time.perf_counter()
    # for i in range(10):
    #     O_dace = func(Q=Q, K=K, V=V, N=N, d=d)
    # finish = time.perf_counter()
    # print("Flash attention DaCe 2 mean execution: ", (finish - start) / 10, "seconds", flush=True)
    # gflops = 4 * N * N * d / ((finish - start)/10) / 1e9
    # print("Flash attention DaCe 2 mean GFLOP/s: ", gflops, flush=True)
    # print()

    # if test_gpu:
    #     sdfg = flash_attention_dace_2.to_sdfg(simplify=False)
    #     sdfg.simplify()
    #     auto_optimize(sdfg, dace.DeviceType.GPU, use_gpu_storage=True)
    #     func = sdfg.compile()
    #     O_dev = func(Q=Q_dev, K=K_dev, V=V_dev, N=N, d=d)
    #     if validate:
    #         O_dace = cp.asnumpy(O_dev)
    #         print(np.linalg.norm(O_std_ref - O_dace) / np.linalg.norm(O_std_ref))
    #         assert np.allclose(O_std_ref, O_dace)
    #     start = time.perf_counter()
    #     for i in range(10):
    #         O_dev = func(Q=Q_dev, K=K_dev, V=V_dev, N=N, d=d)
    #     finish = time.perf_counter()
    #     print("Flash attention DaCe 2 GPU mean execution: ", (finish - start) / 10, "seconds", flush=True)
    #     gflops = 4 * N * N * d / ((finish - start)/10) / 1e9
    #     print("Flash attention DaCe 2 GPU mean GFLOP/s: ", gflops, flush=True)
    #     print()
    
    num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    print("OMP_NUM_THREADS:", num_threads)

    # Ti_val = N // num_threads
    # Tj_val = N // num_threads
    Ti_val = args.Ti
    Tj_val = args.Tj

    # sdfg = flash_attention_dace_3.to_sdfg(simplify=False)
    # sdfg.simplify()
    # auto_optimize(sdfg, dace.DeviceType.CPU)
    # func = sdfg.compile()
    # O_dace = func(Q=Q, K=K, V=V, N=N, d=d, Ti=Ti_val, Tj=Tj_val)
    # if validate:
    #     print(np.linalg.norm(O_std_ref - O_dace) / np.linalg.norm(O_std_ref))
    #     assert np.allclose(O_std_ref, O_dace)
    # start = time.perf_counter()
    # for i in range(10):
    #     O_dace = func(Q=Q, K=K, V=V, N=N, d=d, Ti=Ti_val, Tj=Tj_val)
    # finish = time.perf_counter()
    # print("Flash attention DaCe 3 mean execution: ", (finish - start) / 10, "seconds", flush=True)
    # gflops = 4 * N * N * (d + 5) / ((finish - start)/10) / 1e9
    # print("Flash attention DaCe 3 mean GFLOP/s: ", gflops, flush=True)
    # print()

    sdfg = flash_attention_dace_4.to_sdfg(simplify=False)
    sdfg.simplify()
    auto_optimize(sdfg, dace.DeviceType.CPU)
    func = sdfg.compile()
    O_dace = np.empty_like(Q)
    func(Q=Q, K=K, V=V, N=N, d=d, O=O_dace, Ti=Ti_val, Tj=Tj_val)
    if validate:
        print(np.linalg.norm(O_std_ref - O_dace) / np.linalg.norm(O_std_ref))
        assert np.allclose(O_std_ref, O_dace)
    start = time.perf_counter()
    for i in range(10):
        func(Q=Q, K=K, V=V, O=O_dace, N=N, d=d, Ti=Ti_val, Tj=Tj_val)
    finish = time.perf_counter()
    print("Flash attention DaCe 3 mean execution: ", (finish - start) / 10, "seconds", flush=True)
    gflops = 4 * N * N * (d + 5) / ((finish - start)/10) / 1e9
    print("Flash attention DaCe 3 mean GFLOP/s: ", gflops, flush=True)
    print()

    # if test_gpu:
    #     sdfg = flash_attention_dace_3.to_sdfg(simplify=False)
    #     sdfg.simplify()
    #     auto_optimize(sdfg, dace.DeviceType.GPU, use_gpu_storage=True)
    #     func = sdfg.compile()
    #     O_dev = func(Q=Q_dev, K=K_dev, V=V_dev, N=N, d=d)
    #     if validate:
    #         O_dace = cp.asnumpy(O_dev)
    #         print(np.linalg.norm(O_std_ref - O_dace) / np.linalg.norm(O_std_ref))
    #         assert np.allclose(O_std_ref, O_dace)
    #     start = time.perf_counter()
    #     for i in range(10):
    #         O_dev = func(Q=Q_dev, K=K_dev, V=V_dev, N=N, d=d)
    #     finish = time.perf_counter()
    #     print("Flash attention DaCe 3 GPU mean execution: ", (finish - start) / 10, "seconds", flush=True)
    #     gflops = 4 * N * N * d / ((finish - start)/10) / 1e9
    #     print("Flash attention DaCe 3 GPU mean GFLOP/s: ", gflops, flush=True)
    #     print()
