import dace
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")

from flash_attention_dace import standard_attention_dace, flash_attention_dace_4
from dace.transformation.auto.auto_optimize import auto_optimize


datasizes = [(64, 16384), (64, 32768), (128, 8192), (128, 16384)]
tilesizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]


if __name__ == "__main__":

    # # Standard Attention
    # sa_sdfg = standard_attention_dace.to_sdfg(simplify=False)
    # sa_sdfg.simplify()
    # sa_sdfg.expand_library_nodes()
    # auto_optimize(sa_sdfg, dace.DeviceType.CPU)

    # Flash Attention
    fa_sdfg = flash_attention_dace_4.to_sdfg(simplify=False)
    fa_sdfg.simplify()
    auto_optimize(fa_sdfg, dace.DeviceType.CPU)
    fa_func = fa_sdfg.compile()

    rng = np.random.default_rng(42)

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    for datasize in datasizes:

        # sa_func = sa_sdfg.compile()
        # fa_func = fa_sdfg.compile()

        d, N = datasize
        print(f"\nData size: {d} x {N}")
        Q = rng.random((N, d), dtype=np.float32)
        K = rng.random((N, d), dtype=np.float32)
        V = rng.random((N, d), dtype=np.float32)
        O = np.empty_like(Q)

        # for threads in (1, 2, 4, 8, 16, 32):

        #     print(f"\nThreads: {threads}\n")
        #     os.environ["OMP_NUM_THREADS"] = str(threads)
        #     os.environ["MKL_NUM_THREADS"] = str(threads)
        #     os.environ["OPENBLAS_NUM_THREADS"] = str(threads)

        # O_ref = sa_func(Q=Q, K=K, V=V, N=N, d=d)
        # start = time.perf_counter()
        # for i in range(10):
        #     sa_func(Q=Q, K=K, V=V, N=N, d=d)
        # finish = time.perf_counter()
        # runtime = (finish - start) / 10
        # gflops = (4 * N * N * (d + 5)) / (runtime * 1e9)
        # print(f"Standard attention DaCe mean execution: {runtime} seconds ({gflops} GFLOP/s)", flush=True)

        for tilesize in tilesizes:

            Ti, Tj = tilesize
            # Tj = d

            fa_func(Q=Q, K=K, V=V, O=O, N=N, d=d, Ti=Ti, Tj=Tj)
            start = time.perf_counter()
            for i in range(10):
                fa_func(Q=Q, K=K, V=V, O=O, N=N, d=d, Ti=Ti, Tj=Tj)
            finish = time.perf_counter()
            runtime = (finish - start) / 10
            gflops = (4 * N * N * (d + 5)) / (runtime * 1e9)
            print(f"Flash attention DaCe (Ti={Ti}, Tj={Tj}) mean execution: {runtime} seconds ({gflops} GFLOP/s)", flush=True)
