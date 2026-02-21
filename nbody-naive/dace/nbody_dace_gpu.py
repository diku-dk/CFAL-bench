import argparse
import dace
import cupy as cp
import numpy as np
import numpy.typing as npt
import time

from dace.transformation.dataflow import MapExpansion, MapFusion, TaskletFusion
from timeit import repeat
from nbody_dace_gpu_impl import get_nbody_dace_gpu


eps = np.finfo(np.float64).eps


def nbody_cpython(pos_mass: npt.NDArray[np.float64],
                  vel: npt.NDArray[np.float64],
                  dt: np.float64,
                  iterations: int):

    for _ in range(iterations):

        diff = pos_mass[np.newaxis, :, 0:3] - pos_mass[:, np.newaxis, 0:3]
        dist_sq = np.sum(diff * diff, axis=2)
        inv_dist = 1.0 / np.sqrt(dist_sq + eps)
        inv_dist3 = inv_dist * inv_dist * inv_dist
        accel = np.sum(diff * pos_mass[:, np.newaxis, 3] * inv_dist3[:, :, np.newaxis], axis=1)
        vel[:] += accel * dt
        pos_mass[:, 0:3] += vel * dt


def relerror(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-N", type=int, default=50000)
    argparser.add_argument("-iterations", type=int, default=10)	
    args = vars(argparser.parse_args())

    print(f"N = {args['N']}, iterations = {args['iterations']}", flush=True)

    sdfg = get_nbody_dace_gpu()
    
    rng = np.random.default_rng(0)

    print("\nValidating correctness ...", flush=True)

    num_particles = 1000
    num_iterations = 10
    pos_mass = rng.random((num_particles, 4))
    vel = rng.random((num_particles, 3))
    dt = 0.01

    pos_mass_dace = cp.asarray(pos_mass)
    vel_dace = cp.asarray(vel)
    cp.cuda.runtime.deviceSynchronize()

    nbody_cpython(pos_mass, vel, dt, num_iterations)

    for tb_sz in (32, 64, 128, 256, 512):

        p_tmp = cp.copy(pos_mass_dace)
        v_tmp = cp.copy(vel_dace)

        sdfg.specialize({'tb_sz': tb_sz})
        func = sdfg.compile()
        func(pos_mass=p_tmp, vel=v_tmp, dt=dt, N=num_particles, iterations=num_iterations)
        cp.cuda.runtime.deviceSynchronize()
        print(f'DaCe GPU (thread block size = {tb_sz}) relative error: {relerror(cp.asnumpy(p_tmp), pos_mass)}', flush=True)
        print(f'DaCe GPU (thread block size = {tb_sz}) relative error: {relerror(cp.asnumpy(v_tmp), vel)}', flush=True)

    num_particles = args['N']
    num_iterations = args['iterations']
    pos_mass = rng.random((num_particles, 4))
    pos_mass_gpu = cp.asarray(pos_mass)
    vel = rng.random((num_particles, 3))
    vel_gpu = cp.asarray(vel)
    dt = 0.01

    print('\nBenchmarking ...', flush=True)
    print()

    num_warmup = 2

    for tb_sz in (32, 64, 128, 256, 512):

        sdfg.specialize({'tb_sz': tb_sz})
        func = sdfg.compile()
        p_tmp = cp.copy(pos_mass_gpu)
        v_tmp = cp.copy(vel_gpu)

        def _func():
            func(pos_mass=p_tmp, vel=v_tmp, dt=dt, N=num_particles, iterations=num_iterations)
            cp.cuda.runtime.deviceSynchronize()
        
        for _ in range(num_warmup):
            _func()
        runtimes = repeat(lambda: _func(), number=1, repeat=10)
        mean = np.mean(runtimes)
        std = np.std(runtimes)
        repeatitions = 10
        while std > 0.01 * mean and repeatitions < 200:
            print(f"Standard deviation too high ({std * 100 / mean:.2f}% of the mean) after {repeatitions} repeatitions ...", flush=True)
            runtimes.extend(repeat(lambda: _func(), number=1, repeat=10))
            mean = np.mean(runtimes)
            std = np.std(runtimes)
            repeatitions += 10
        flops = (19.0 * num_particles * num_particles + 12.0 * num_particles) * num_iterations / (1e9 * mean)
        print(f"DaCe GPU (thread block size = {tb_sz}) runtime: mean {mean} s ({flops} Gflop/s), std {std * 100 / mean:.2f}%", flush=True)

        with open(f"nbody_dace_gpu_tb{tb_sz}_{num_particles}_{num_iterations}.txt", "w") as fp:
            for t in runtimes:
                fp.write(f"{t}\n")
