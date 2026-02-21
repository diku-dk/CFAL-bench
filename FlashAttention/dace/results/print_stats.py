import itertools
import numpy as np
import scipy as sp
import os


dir_path = os.path.dirname(os.path.realpath(__file__))


sizes = [(64, 16384), (64, 32768), (128, 8192), (128, 16384)]
cpu_tilesizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]
gpu_tb_sizes = [64, 128]
gpu_tilesizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def time2gflops(d, N, runtimes):
    return (N * N * (4 * d + 5)) / (runtimes * 1e9)


def print_stats(d, N, runtimes):

    mean_runtime = np.mean(runtimes)
    std_pop_runtime = np.std(runtimes, ddof=0)
    std_samp_runtime = np.std(runtimes, ddof=1)
    ci_95_runtime = sp.stats.bootstrap((runtimes,), np.mean, confidence_level=0.95).confidence_interval
    ci_99_runtime = sp.stats.bootstrap((runtimes,), np.mean, confidence_level=0.99).confidence_interval

    print(f"        Runtime: ({runtimes.size} runs)", flush=True)
    print(f"            mean {mean_runtime:.6f} s", flush=True)
    print(f"            std (population) {std_pop_runtime:.6f} s ({std_pop_runtime/mean_runtime*100:.2f}%)", flush=True)
    print(f"            std (sample) {std_samp_runtime:.6f} s ({std_samp_runtime/mean_runtime*100:.2f}%)", flush=True)
    print(f"            95% CI [{ci_95_runtime.low:.6f}, {ci_95_runtime.high:.6f}]", flush=True)
    print(f"            99% CI [{ci_99_runtime.low:.6f}, {ci_99_runtime.high:.6f}]", flush=True)

    gflops = time2gflops(d, N, runtimes)
    mean_gflops = np.mean(gflops)
    std_pop_gflops = np.std(gflops, ddof=0)
    std_samp_gflops = np.std(gflops, ddof=1)
    ci_95_gflops = sp.stats.bootstrap((gflops,), np.mean, confidence_level=0.95).confidence_interval
    ci_99_gflops = sp.stats.bootstrap((gflops,), np.mean, confidence_level=0.99).confidence_interval

    print(f"        Gflop/s: ({gflops.size} runs)", flush=True)
    print(f"            mean {mean_gflops:.6f} Gflop/s", flush=True)
    print(f"            std (population) {std_pop_gflops:.6f} Gflop/s ({std_pop_gflops/mean_gflops*100:.2f}%)", flush=True)
    print(f"            std (sample) {std_samp_gflops:.6f} Gflop/s ({std_samp_gflops/mean_gflops*100:.2f}%)", flush=True)
    print(f"            95% CI [{ci_95_gflops.low:.6f}, {ci_95_gflops.high:.6f}]", flush=True)
    print(f"            99% CI [{ci_99_gflops.low:.6f}, {ci_99_gflops.high:.6f}]", flush=True)


if __name__ == "__main__":

    for size in sizes:

        d, N = size
        print(f"d = {d}, N = {N}\n", flush=True)

        # CPU single core
        best_mean = np.inf
        best_ti_tj = None
        for Ti, Tj in cpu_tilesizes:
            with open(os.path.join(dir_path, f"fa_dace_cpu_1_Ti{Ti}_Tj{Tj}_{d}_{N}.txt"), "r") as fp:
                runtimes = np.array([float(line.strip()) for line in fp])
                mean_runtime = np.mean(runtimes)
                if mean_runtime < best_mean:
                    best_mean = mean_runtime
                    best_ti_tj = (Ti, Tj)
        print(f"    CPU single core (best tile size = {best_ti_tj}):", flush=True)
        with open(os.path.join(dir_path, f"fa_dace_cpu_1_Ti{best_ti_tj[0]}_Tj{best_ti_tj[1]}_{d}_{N}.txt"), "r") as fp:
            runtimes = np.array([float(line.strip()) for line in fp])
            print_stats(d, N, runtimes)
            print("", flush=True)

        # CPU multi-core
        best_mean = np.inf
        best_ti_tj = None
        for Ti, Tj in cpu_tilesizes:
            with open(os.path.join(dir_path, f"fa_dace_cpu_32_Ti{Ti}_Tj{Tj}_{d}_{N}.txt"), "r") as fp:
                runtimes = np.array([float(line.strip()) for line in fp])
                mean_runtime = np.mean(runtimes)
                if mean_runtime < best_mean:
                    best_mean = mean_runtime
                    best_ti_tj = (Ti, Tj)
        print(f"    CPU multi-core (best tile size = {best_ti_tj}):", flush=True)
        with open(os.path.join(dir_path, f"fa_dace_cpu_32_Ti{best_ti_tj[0]}_Tj{best_ti_tj[1]}_{d}_{N}.txt"), "r") as fp:
            runtimes = np.array([float(line.strip()) for line in fp])
            print_stats(d, N, runtimes)
            print("", flush=True)

        # GPU
        best_mean = np.inf
        best_tbsz_ti = None
        for tbsz, Ti in itertools.product(gpu_tb_sizes, gpu_tilesizes):
            with open(os.path.join(dir_path, f"fa_dace_gpu_tb{tbsz}_Ti{Ti}_{d}_{N}.txt"), "r") as fp:
                runtimes = np.array([float(line.strip()) for line in fp])
                mean_runtime = np.mean(runtimes)
                if mean_runtime < best_mean:
                    best_mean = mean_runtime
                    best_tbsz_ti = (tbsz, Ti)
        print(f"    GPU (best TB and tile sizes = {best_tbsz_ti}):", flush=True)
        with open(os.path.join(dir_path, f"fa_dace_gpu_tb{best_tbsz_ti[0]}_Ti{best_tbsz_ti[1]}_{d}_{N}.txt"), "r") as fp:
            runtimes = np.array([float(line.strip()) for line in fp])
            print_stats(d, N, runtimes)
            print("", flush=True)
