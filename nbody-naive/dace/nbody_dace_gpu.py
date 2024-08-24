import argparse
import dace
import cupy as cp
import numpy as np
import numpy.typing as npt

from dace.transformation.dataflow import MapFusion, TaskletFusion
from timeit import repeat


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


N, iterations, tb_sz = (dace.symbol(s) for s in ('N', 'iterations', 'tb_sz'))

@dace.program
def nbody_dace_gpu(pos_mass: dace.float64[N, 4] @ dace.StorageType.GPU_Global,
                   vel: dace.float64[N, 3] @ dace.StorageType.GPU_Global,
                   dt: dace.float64):
    
    for _ in range(iterations):

        for block_start in dace.map[0:N:tb_sz] @ dace.ScheduleType.GPU_Device:
            for tid in dace.map[0:tb_sz] @ dace.ScheduleType.GPU_ThreadBlock:
                pos_mass_shared = dace.define_local((tb_sz, 4), dtype=np.float64, storage=dace.StorageType.GPU_Shared)
                accel = dace.define_local([3], dtype=dace.float64, storage=dace.StorageType.Register)
                accel[:] = 0.0
                pos_mass_i = pos_mass[block_start + tid]
                for j in range(0, N, tb_sz):
                    if j + tid < N:
                        pos_mass_shared[tid, 0:4] = pos_mass[j + tid, 0:4]
                    with dace.tasklet(side_effects=True):
                        __syncthreads()
                    for k in range(min(tb_sz, N - j)):
                        with dace.tasklet:
                            posi << pos_mass_i[0:3]
                            posj << pos_mass_shared[k, 0:3]
                            massj << pos_mass_shared[k, 3]
                            accel_in << accel
                            accel_out >> accel
                            diff_x = posj[0] - posi[0]
                            diff_y = posj[1] - posi[1]
                            diff_z = posj[2] - posi[2]
                            dist_sq = diff_x*diff_x + diff_y*diff_y + diff_z*diff_z
                            inv_dist = 1.0 / dace.math.sqrt(dist_sq + eps)
                            inv_dist3 = inv_dist * inv_dist * inv_dist * massj
                            accel_out[0] = accel_in[0] + diff_x * inv_dist3
                            accel_out[1] = accel_in[1] + diff_y * inv_dist3
                            accel_out[2] = accel_in[2] + diff_z * inv_dist3
                    with dace.tasklet(side_effects=True):
                        __syncthreads()
                if block_start + tid < N:
                    vel[block_start + tid, 0:3] += dt * accel

        pos_mass[:, 0:3] += vel * dt


def relerror(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-N", type=int, default=50000)
    argparser.add_argument("-iterations", type=int, default=10)	
    args = vars(argparser.parse_args())

    num_particles = 1000
    num_iterations = 10

    rng = np.random.default_rng(0)
    pos_mass = rng.random((num_particles, 4))
    pos_mass_gpu = cp.asarray(pos_mass)
    vel = rng.random((num_particles, 3))
    vel_gpu = cp.asarray(vel)
    dt = 0.01

    pos_mass_dace = cp.asarray(pos_mass)
    vel_dace = cp.asarray(vel)

    nbody_cpython(pos_mass, vel, dt, num_iterations)

    sdfg = nbody_dace_gpu.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated([MapFusion, TaskletFusion])
    sdfg.simplify()

    print("\nValidating correctness ...", flush=True)

    for tb_sz in (32, 64, 128, 256, 512):

        p_tmp = cp.copy(pos_mass_gpu)
        v_tmp = cp.copy(vel_gpu)

        sdfg.specialize({'tb_sz': tb_sz})
        sdfg(pos_mass=p_tmp, vel=v_tmp, dt=dt, N=num_particles, iterations=num_iterations)
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
        while std > 0.01 * mean:
            print(f"Standard deviation too high ({std * 100 / mean:.2f}% of the mean) after {repeatitions} repeatitions ...", flush=True)
            runtimes.extend(repeat(lambda: _func(), number=1, repeat=10))
            mean = np.mean(runtimes)
            std = np.std(runtimes)
            repeatitions += 10
        flops = (19.0 * num_particles * num_particles + 12.0 * num_particles) * num_iterations / (1e9 * mean)
        print(f"DaCe GPU (thread block size = {tb_sz}) runtime: mean {mean} s ({flops} Gflop/s), std {std * 100 / mean:.2f}%", flush=True)
