import cupy as cp
import dace
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")

from timeit import repeat
from flash_attention_dace_gpu import custom_attention_dace
from dace.transformation.auto.auto_optimize import auto_optimize, apply_gpu_storage


datasizes = [(64, 16384), (64, 32768), (128, 8192), (128, 16384)]
tilesizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


if __name__ == "__main__":

    # Flash Attention
    sdfg = custom_attention_dace.to_sdfg(simplify=False)
    apply_gpu_storage(sdfg)
    for sd in sdfg.all_sdfgs_recursive():
        if sd.parent_sdfg is not None and sd.parent_sdfg is sdfg:
            sd.simplify()
            auto_optimize(sd, dace.DeviceType.GPU, use_gpu_storage=True)
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.schedule = dace.ScheduleType.Sequential
    sdfg.simplify()
    fa_func = sdfg.compile()

    rng = np.random.default_rng(42)

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    for datasize in datasizes:

        d, N = datasize
        print(f"\nData size: {d} x {N}")
        Q = cp.asarray(rng.random((N, d), dtype=np.float32))
        K = cp.asarray(rng.random((N, d), dtype=np.float32))
        V = cp.asarray(rng.random((N, d), dtype=np.float32))
        O = cp.empty_like(Q)

        for tilesize in tilesizes:

            Ti = tilesize

            num_warmup = 2

            def _func():
                fa_func(Q=Q, K=K, V=V, O=O, N=N, d=d, Ti=Ti)
            
            for _ in range(num_warmup):
                _func()
            runtimes = repeat(lambda: _func(), number=1, repeat=10)
            mean = np.mean(runtimes)
            std = np.std(runtimes)
            repeatitions = 10
            while std > 0.01 * mean and repeatitions < 100:
                print(f"Standard deviation too high ({std * 100 / mean:.2f}% of the mean) after {repeatitions} repeatitions ...", flush=True)
                runtimes.extend(repeat(lambda: _func(), number=1, repeat=10))
                mean = np.mean(runtimes)
                std = np.std(runtimes)
                repeatitions += 10
            flops = (N * N * (4 * d + 5)) / (mean * 1e9)
            print(f"DaCe GPU runtime (Ti={Ti}): mean {mean} s ({flops} Gflop/s), std {std * 100 / mean:.2f}%", flush=True)
