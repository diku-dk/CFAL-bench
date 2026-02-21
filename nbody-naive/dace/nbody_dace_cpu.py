import argparse
import dace
import numpy as np
import numpy.typing as npt
import time
import os
from timeit import repeat

from nbody_dace_cpu_impl import get_nbody_dace_cpu


from dace.sdfg import utils as sdutil
from dace.transformation.auto.auto_optimize import auto_optimize, greedy_fuse
from dace.transformation.dataflow import MapExpansion, MapFusion, TaskletFusion
from dace.transformation.interstate import HoistState


eps = np.finfo(np.float64).eps


def nbody_cpython(pos_mass: npt.NDArray[np.float64],
                  vel: npt.NDArray[np.float64],
                  dt: np.float64,
                  iterations: int):

    for _ in range(iterations):

        dist = pos_mass[0:3, np.newaxis, :] - pos_mass[0:3, :, np.newaxis]
        dist_sq = np.sum(dist * dist, axis=0)
        inv_dist = 1.0 / np.sqrt(dist_sq + eps)
        inv_dist3 = inv_dist * inv_dist * inv_dist
        accel = np.sum(dist * pos_mass[3, np.newaxis, :] * inv_dist3[np.newaxis, :, :], axis=2)
        vel[:] = vel + dt * accel
        pos_mass[0:3] = pos_mass[0:3] + dt * vel

N = dace.symbol("N")
iterations = dace.symbol("iterations")

@dace.program
def nbody_dace_cpu(pos_mass: dace.float64[4, N],
                   vel: dace.float64[3, N],
                   dt: dace.float64):

    for _ in range(iterations):

        dist = pos_mass[0:3, np.newaxis, :] - pos_mass[0:3, :, np.newaxis]
        dist_sq = np.sum(dist * dist, axis=0)
        inv_dist = 1.0 / np.sqrt(dist_sq + eps)
        inv_dist3 = inv_dist * inv_dist * inv_dist * pos_mass[3]
        accel = np.sum(dist * inv_dist3[np.newaxis, :, :], axis=2)
        vel[:] = vel + dt * accel
        pos_mass[0:3] = pos_mass[0:3] + dt * vel


def relerror(val, ref):
    return np.linalg.norm(val - ref) / np.linalg.norm(ref)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-N", type=int, default=50000)
    argparser.add_argument("-iterations", type=int, default=10)	
    args = vars(argparser.parse_args())

    omp_num_threads = os.getenv("OMP_NUM_THREADS", "1")

    print(f"N = {args['N']}, iterations = {args['iterations']}, OMP_NUM_THREADS = {omp_num_threads}", flush=True)

    num_particles = 1000
    num_iterations = 10

    rng = np.random.default_rng(0)
    pos_mass = rng.random((4, num_particles))
    vel = rng.random((3, num_particles))
    dt = 0.01

    pos_mass_dace = np.asarray(pos_mass)
    vel_dace = np.asarray(vel)

    nbody_cpython(pos_mass, vel, dt, num_iterations)

    sdfg = get_nbody_dace_cpu()

    print("\nValidating correctness ...", flush=True)

    sdfg(pos_mass=pos_mass_dace, vel=vel_dace, dt=dt, N=num_particles, iterations=num_iterations)
    print(f'DaCe CPU relative error: {relerror(pos_mass_dace, pos_mass)}', flush=True)
    print(f'DaCe CPU relative error: {relerror(vel_dace, vel)}', flush=True)

    num_particles = args['N']
    num_iterations = args['iterations']
    pos_mass = rng.random((4, num_particles))
    vel = rng.random((3, num_particles))
    dt = 0.01

    print('\nBenchmarking ...', flush=True)
    print()

    num_warmup = 2

    t0 = time.perf_counter()
    _ = sdfg.generate_code()
    t1 = time.perf_counter()
    print(f"Generating code from SDFG took {t1 - t0:.6f} seconds", flush=True)
    func = sdfg.compile()
    t2 = time.perf_counter()
    print(f"Compiling SDFG code took {t2 -  t1:.6f} seconds", flush=True)

    def _func():
        func(pos_mass=pos_mass, vel=vel, dt=dt, N=num_particles, iterations=num_iterations)
    
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
    print(f"DaCe CPU runtime: mean {mean} s ({flops} Gflop/s), std {std * 100 / mean:.2f}%", flush=True)

    with open(f"nbody_dace_cpu_{omp_num_threads}_{num_particles}_{num_iterations}.txt", "w") as fp:
        for t in runtimes:
            fp.write(f"{t}\n")

