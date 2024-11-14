import dace
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")

from timeit import repeat
# from flash_attention_dace_cpu import flash_attention_dace_4
from flash_attention_dace_cpu import get_flash_attention_dace_cpu
from dace.transformation.auto.auto_optimize import auto_optimize


datasizes = [(64, 16384), (64, 32768), (128, 8192), (128, 16384)]
tilesizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]


if __name__ == "__main__":

    # # Flash Attention
    # fa_sdfg = flash_attention_dace_4.to_sdfg(simplify=False)
    # fa_sdfg.simplify()
    # auto_optimize(fa_sdfg, dace.DeviceType.CPU)
    # fa_func = fa_sdfg.compile()
    fa_func = get_flash_attention_dace_cpu()

    rng = np.random.default_rng(42)

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    for datasize in datasizes:

        d, N = datasize
        print(f"\nData size: {d} x {N}")
        Q = rng.random((N, d), dtype=np.float32)
        K = rng.random((N, d), dtype=np.float32)
        V = rng.random((N, d), dtype=np.float32)
        O = np.empty_like(Q)

        for tilesize in tilesizes:

            Ti, Tj = tilesize

            num_warmup = 2

            def _func():
                fa_func(Q=Q, K=K, V=V, O=O, N=N, d=d, Ti=Ti, Tj=Tj)
            
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
            print(f"DaCe CPU runtime (Ti={Ti}, Tj={Tj}): mean {mean} s ({flops} Gflop/s), std {std * 100 / mean:.2f}%", flush=True)
