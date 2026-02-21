import numpy as np
import scipy as sp
import os


dir_path = os.path.dirname(os.path.realpath(__file__))


sizes = [(1000, 100000), (10000, 1000), (100000, 10)]


def time2gflops(N, iterations, runtimes):
    return (19.0 * N * N + 12.0 * N) * iterations / (1e9 * runtimes)


def print_stats(N, iterations, runtimes):

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

    gflops = time2gflops(N, iterations, runtimes)
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

        N, iterations = size
        print(f"N = {N}, iterations = {iterations}\n", flush=True)

        # CPU single core
        print("    CPU single core:", flush=True)
        with open(os.path.join(dir_path, f"nbody_dace_cpu_1_{N}_{iterations}.txt"), "r") as fp:
            runtimes = np.array([float(line.strip()) for line in fp])
            print_stats(N, iterations, runtimes)
            print("", flush=True)

        # CPU multi-core
        print("    CPU multi-core:", flush=True)
        with open(os.path.join(dir_path, f"nbody_dace_cpu_32_{N}_{iterations}.txt"), "r") as fp:
            runtimes = np.array([float(line.strip()) for line in fp])
            print_stats(N, iterations, runtimes)
            print("", flush=True)

        # GPU
        best_mean = np.inf
        best_tb_sz = None
        for tb_sz in (32, 64, 128, 256, 512):
            with open(os.path.join(dir_path, f"nbody_dace_gpu_tb{tb_sz}_{N}_{iterations}.txt"), "r") as fp:
                runtimes = np.array([float(line.strip()) for line in fp])
                mean_runtime = np.mean(runtimes)
                if mean_runtime < best_mean:
                    best_mean = mean_runtime
                    best_tb_sz = tb_sz
        print(f"    GPU (best thread block size = {best_tb_sz}):", flush=True)
        with open(os.path.join(dir_path, f"nbody_dace_gpu_tb{best_tb_sz}_{N}_{iterations}.txt"), "r") as fp:
            runtimes = np.array([float(line.strip()) for line in fp])
            print_stats(N, iterations, runtimes)
            print("", flush=True)
