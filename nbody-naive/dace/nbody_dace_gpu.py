import dace
import cupy as cp
import cupyx as cpx
import numpy as np
import numpy.typing as npt

from timeit import repeat
from cupyx.profiler import benchmark


def nbody_python(pos: npt.NDArray[np.float64],
                 vel: npt.NDArray[np.float64],
                 mass: npt.NDArray[np.float64],
                 dt: np.float64,
                 iterations: int):

    for _ in range(iterations):

        dist =  pos[np.newaxis, :] - pos[:, np.newaxis]
        inv_dist = 1.0 / np.sqrt(np.sum(dist**2, axis=2) + 1e-9)**3
        f = dist * mass[np.newaxis, :, np.newaxis] * inv_dist[:, :, np.newaxis]
        f_red = np.sum(f, axis=1)
        vel += dt * f_red
        pos += dt * vel


kernel_code = '''
extern "C" __global__ void accelerate(const double4* pos_mass, double3*  accel, int N) {{

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int start_row = bid * num_threads;
    int end_row = min(start_row + num_threads, N);
    int actual_threads = end_row - start_row;

    __shared__ double4 pos_j[{max_threads}];

    double3 acc = make_double3(0.0, 0.0, 0.0);
    double3 dist;
    double4 pos_i;

    if (tid < actual_threads) {{
        pos_i = pos_mass[start_row + tid];
    }}

    for (int j = 0; j < N; j += num_threads) {{
    
        if (j + tid < N) {{
            pos_j[tid] = pos_mass[j + tid];
        }}
        __syncthreads();

        if (tid < actual_threads) {{
        
            /*
            int max_k = min(num_threads, N - j);
            int tile_k_num = (max_k + {unroll_factor} - 1) / {unroll_factor};
        
            for (int tile_k = 0; tile_k < tile_k_num - 1; tile_k++) {{
                #pragma unroll {unroll_factor}
                for (int k = tile_k * {unroll_factor}; k < (tile_k + 1) * {unroll_factor}; k++) {{
                    double4 pos_j_k = pos_j[k];
                    dist.x = pos_j_k.x - pos_i.x;
                    dist.y = pos_j_k.y - pos_i.y;
                    dist.z = pos_j_k.z - pos_i.z;
                    double dist_sq = dist.x*dist.x + dist.y*dist.y + dist.z*dist.z;
                    double inv_dist = 1.0 / sqrt(dist_sq + 1e-9);
                    double inv_dist3 = inv_dist*inv_dist*inv_dist;
                    acc.x += dist.x * pos_j_k.w * inv_dist3;
                    acc.y += dist.y * pos_j_k.w * inv_dist3;
                    acc.z += dist.z * pos_j_k.w * inv_dist3;
                }}
            }}

            for (int k = (tile_k_num - 1) * {unroll_factor}; k < min(num_threads, N - j); k++) {{
                double4 pos_j_k = pos_j[k];
                dist.x = pos_j_k.x - pos_i.x;
                dist.y = pos_j_k.y - pos_i.y;
                dist.z = pos_j_k.z - pos_i.z;
                double dist_sq = dist.x*dist.x + dist.y*dist.y + dist.z*dist.z;
                double inv_dist = 1.0 / sqrt(dist_sq + 1e-9);
                double inv_dist3 = inv_dist*inv_dist*inv_dist;
                acc.x += dist.x * pos_j_k.w * inv_dist3;
                acc.y += dist.y * pos_j_k.w * inv_dist3;
                acc.z += dist.z * pos_j_k.w * inv_dist3;
            }}
            */

            for (int k = 0; k < min(num_threads, N - j); k++) {{
                double4 pos_j_k = pos_j[k];
                dist.x = pos_j_k.x - pos_i.x;
                dist.y = pos_j_k.y - pos_i.y;
                dist.z = pos_j_k.z - pos_i.z;
                double dist_sq = dist.x*dist.x + dist.y*dist.y + dist.z*dist.z;
                double inv_dist = 1.0 / sqrt(dist_sq + 1e-9);
                double inv_dist3 = inv_dist*inv_dist*inv_dist;
                acc.x += dist.x * pos_j_k.w * inv_dist3;
                acc.y += dist.y * pos_j_k.w * inv_dist3;
                acc.z += dist.z * pos_j_k.w * inv_dist3;

                /*
                if (bid + tid == 0) {{
                    printf("i=%d, j=%d, k=%d, dist=(%f, %f, %f), dist_sq=%f, inv_dist=%f, inv_dist3=%f, acc=(%f, %f, %f)\\n", start_row + tid, j, k, dist.x, dist.y, dist.z, dist_sq, inv_dist, inv_dist3, acc.x, acc.y, acc.z);
                }}
                */
            }}
            
        }}

        __syncthreads();
    }}

    if (tid < actual_threads) {{
        accel[start_row + tid] = acc;
    }}
}}

extern "C" __global__ void update(double4* pos, double3* vel, double3* accel, double dt, int N) {{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {{
        double4 cpos = pos[i];
        double3 cvel = vel[i];
        double3 caccel = accel[i];
        double3 nvel = make_double3(cvel.x + dt * caccel.x,
                                    cvel.y + dt * caccel.y,
                                    cvel.z + dt * caccel.z);
        double4 npos = make_double4(cpos.x + dt * nvel.x,
                                    cpos.y + dt * nvel.y,
                                    cpos.z + dt * nvel.z,
                                    cpos.w);
        pos[i] = npos;
        vel[i] = nvel;
    }}
}}
'''

def nbody_cupy(pos_mass, vel, dt, iterations, accelerate_kernel, update_kernel, max_threads=256):

    N = np.int32(len(pos_mass))
    threads = min(N, max_threads)
    accel = cp.empty((N, 3), dtype=np.float64)

    for _ in range(iterations):

        num_threads = threads
        num_blocks = (N + num_threads - 1) // num_threads
        accelerate_kernel((num_blocks,),  (num_threads,), (pos_mass, accel, N))

        num_threads = threads
        num_blocks = (N + num_threads - 1) // num_threads
        update_kernel((num_blocks,), (num_threads,), (pos_mass, vel, accel, dt, N))
    
    cp.cuda.runtime.deviceSynchronize()

N, iterations, max_threads = dace.symbol('N'), dace.symbol('iterations'), dace.symbol('max_threads')


@dace.program
def nbody_dace_opt_gpu(#pos: dace.float64[N, 3] @ dace.StorageType.GPU_Global,
                        pos_mass: dace.float64[N, 4] @ dace.StorageType.GPU_Global,
                        vel: dace.float64[N, 3] @ dace.StorageType.GPU_Global,
                        # mass: dace.float64[N] @ dace.StorageType.GPU_Global,
                        # accel: dace.float64[N, 3] @ dace.StorageType.GPU_Global,
                        dt: dace.float64):
    
    accel = cp.empty((N, 3), dtype=np.float64)

    for _ in range(iterations):

        for start_i in dace.map[0:N:max_threads] @ dace.ScheduleType.GPU_Device:
            for tid in dace.map[0:max_threads] @ dace.ScheduleType.GPU_ThreadBlock:
                posj = dace.define_local((max_threads, 4), dtype=np.float64, storage=dace.StorageType.GPU_Shared)
                tmp = dace.define_local([3], dtype=dace.float64, storage=dace.StorageType.Register)
                tmp[:] = 0.0
                posi = pos_mass[start_i + tid]
                for j in dace.map[0:N:max_threads] @ dace.ScheduleType.Sequential:
                    if j + tid < N:
                        posj[tid, 0:4] = pos_mass[j + tid, 0:4]
                    with dace.tasklet(side_effects=True):
                        __syncthreads()
                    for k in range(min(max_threads, N - j)):
                        with dace.tasklet:
                            pos_i << posi[0:3]
                            pos_j << posj[k, 0:3]
                            mass_in << posj[k, 3]
                            tmp_in << tmp[0:3]
                            tmp_out >> tmp[0:3]
                            dist0 = pos_j[0] - pos_i[0]
                            dist1 = pos_j[1] - pos_i[1]
                            dist2 = pos_j[2] - pos_i[2]
                            dist_sq = dist0*dist0 + dist1*dist1 + dist2*dist2
                            inv_dist = 1.0 / dace.math.sqrt(dist_sq + 1e-9)
                            inv_dist3 = inv_dist * inv_dist * inv_dist
                            tmp_out[0] = tmp_in[0] + dist0 * mass_in * inv_dist3
                            tmp_out[1] = tmp_in[1] + dist1 * mass_in * inv_dist3
                            tmp_out[2] = tmp_in[2] + dist2 * mass_in * inv_dist3
                    with dace.tasklet(side_effects=True):
                        __syncthreads()
                if start_i + tid < N:
                    accel[start_i + tid, 0:3] = tmp

        # for i in dace.map[0:N] @ dace.ScheduleType.GPU_Device:
        #     with dace.tasklet:
        #         acc_in << accel[i, 0:3]
        #         dt_in << dt
        #         pos_in << pos_mass[i, 0:3]
        #         vel_in << vel[i, 0:3]
        #         pos_out >> pos_mass[i, 0:3]
        #         vel_out >> vel[i, 0:3]
        #         tmp0 = vel_in[0] + dt_in * acc_in[0]
        #         tmp1 = vel_in[1] + dt_in * acc_in[1]
        #         tmp2 = vel_in[2] + dt_in * acc_in[2]
        #         pos_out[0] = pos_in[0] + dt_in * tmp0
        #         pos_out[1] = pos_in[1] + dt_in * tmp1
        #         pos_out[2] = pos_in[2] + dt_in * tmp2
        #         vel_out[0] = tmp0
        #         vel_out[1] = tmp1
        #         vel_out[2] = tmp2

        for gi in dace.map[0:N:max_threads] @ dace.ScheduleType.GPU_Device:
            for i in dace.map[gi:gi+max_threads] @ dace.ScheduleType.GPU_ThreadBlock:
                if i < N:
                    with dace.tasklet:
                        acc_in << accel[i, 0:3]
                        dt_in << dt
                        pos_in << pos_mass[i, 0:3]
                        vel_in << vel[i, 0:3]
                        pos_out >> pos_mass[i, 0:3]
                        vel_out >> vel[i, 0:3]
                        tmp0 = vel_in[0] + dt_in * acc_in[0]
                        tmp1 = vel_in[1] + dt_in * acc_in[1]
                        tmp2 = vel_in[2] + dt_in * acc_in[2]
                        pos_out[0] = pos_in[0] + dt_in * tmp0
                        pos_out[1] = pos_in[1] + dt_in * tmp1
                        pos_out[2] = pos_in[2] + dt_in * tmp2
                        vel_out[0] = tmp0
                        vel_out[1] = tmp1
                        vel_out[2] = tmp2


def relerror(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


if __name__ == '__main__':


    rng = np.random.default_rng(0)
    num_particles = 100
    num_iterations = 10
    pos_mass = rng.random((num_particles, 4))
    pos_mass_gpu = cp.asarray(pos_mass)
    vel = rng.random((num_particles, 3))
    vel_gpu = cp.asarray(vel)
    pos = np.empty((num_particles, 3), dtype=np.float64)
    pos[:] = pos_mass[:, :3]
    mass = np.empty((num_particles,), dtype=np.float64)
    mass[:] = pos_mass[:, 3]
    # pos_dace = cp.asarray(pos)
    pos_mass_dace = cp.asarray(pos_mass)
    # mass_dace = cp.asarray(mass)
    vel_dace = cp.asarray(vel)
    dt = 0.01

    nbody_python(pos_mass[:, :3], vel, pos_mass[:, 3], dt, num_iterations)

    sdfg_gpu = nbody_dace_opt_gpu.to_sdfg(simplify=True)

    p_tmp2 = cp.copy(pos_mass_dace)
    v_tmp2 = cp.copy(vel_dace)


    # for max_threads in (32, 64, 128, 256, 512, 1024):
    for max_threads in (64,):

        p_tmp2[:] = pos_mass_dace
        v_tmp2[:] = vel_dace

        sdfg_gpu.specialize({'max_threads': max_threads})
        # accel_dace = cp.empty((num_particles, 3), dtype=np.float64)
        sdfg_gpu(pos_mass=p_tmp2, vel=v_tmp2, dt=dt, N=num_particles, iterations=num_iterations)
        print(f'DaCe GPU max_threads={max_threads}: {np.allclose(cp.asnumpy(p_tmp2[:, :3]), pos_mass[:, :3])}')
        print(f'DaCe GPU max_threads={max_threads}: {np.allclose(cp.asnumpy(v_tmp2), vel)}')
        print(relerror(cp.asnumpy(p_tmp2[:, :3]), pos_mass[:, :3]))
        print(relerror(cp.asnumpy(v_tmp2), vel))

        code = kernel_code.format(max_threads=max_threads, unroll_factor=4)
        accelerate_kernel = cp.RawKernel(code, 'accelerate')
        update_kernel = cp.RawKernel(code, 'update')
        p_tmp = pos_mass_gpu.copy()
        v_tmp = vel_gpu.copy()
        nbody_cupy(p_tmp, v_tmp, dt, num_iterations, accelerate_kernel, update_kernel, max_threads)

        # # for i in range(num_particles):
        # #     if not cp.allclose(accel_dace[i], accel_cupy[i]):
        # #         print(f'Error at particle {i}')
        # #         print(f'accel_dace[{i}] = {accel_dace[i]}')
        # #         print(f'accel_cupy[{i}] = {accel_cupy[i]}')
        # #         print()
        # print(cp.allclose(accel_dace, accel_cupy))
        print(f'CuPy GPU max_threads={max_threads}: {np.allclose(cp.asnumpy(p_tmp[:, :3]), pos_mass[:, :3])}')
        print(f'CuPy GPU max_threads={max_threads}: {np.allclose(cp.asnumpy(v_tmp), vel)}')
        print()
    # exit(0)

    

    num_particles = 10000
    num_iterations = 10
    pos_mass = rng.random((num_particles, 4))
    pos_mass_gpu = cp.asarray(pos_mass)
    vel = rng.random((num_particles, 3))
    vel_gpu = cp.asarray(vel)
    pos = np.empty((num_particles, 3), dtype=np.float64)
    pos[:] = pos_mass[:, :3]
    pos_gpu = cp.asarray(pos)
    mass = np.empty((num_particles,), dtype=np.float64)
    mass[:] = pos_mass[:, 3]
    mass_gpu = cp.asarray(mass)
    dt = 0.01

    print('Benchmarking...')
    print()

    # for max_threads in (32, 50, 64, 100, 128, 200, 256, 500, 512, 1000, 1024):
    for max_threads in (64,):

        sdfg_gpu.specialize({'max_threads': max_threads})
        accel_dace = cp.empty((num_particles, 3), dtype=np.float64)
        func = sdfg_gpu.compile()
        p_tmp = cp.copy(pos_mass_gpu)
        v_tmp = cp.copy(vel_gpu)

        def _func():
            func(pos_mass=p_tmp, vel=v_tmp, dt=dt, N=num_particles, iterations=num_iterations)
            cp.cuda.runtime.deviceSynchronize()

        runtimes = repeat(lambda: _func(), number=1, repeat=100)
        median_time = np.median(runtimes)
        min_time = np.min(runtimes)
        flops = (18.0 * num_particles * num_particles + 12.0 * num_particles) * num_iterations / (1e9 * median_time)
        print(f"(Num Threads: {max_threads}) Runtime: median {median_time} s ({flops} Gflop/s), min {min_time} s", flush=True)


        # for unroll_factor in (1, 2, 4, 8, 16):
        code = kernel_code.format(max_threads=max_threads, unroll_factor=4)
        accelerate_kernel = cp.RawKernel(code, 'accelerate')
        update_kernel = cp.RawKernel(code, 'update')
        p_tmp = pos_mass_gpu.copy()
        v_tmp = vel_gpu.copy()
        res = benchmark(nbody_cupy, (p_tmp, v_tmp, dt, num_iterations, accelerate_kernel, update_kernel, max_threads), n_repeat=100)
        runtimes = res.gpu_times
        median_time = np.median(runtimes)
        min_time = np.min(runtimes)
        flops = (18.0 * num_particles * num_particles + 12.0 * num_particles) * num_iterations / (1e9 * median_time)
        print(f"(Num Threads: {max_threads}) Runtime: median {median_time} s ({flops} Gflop/s), min {min_time} s", flush=True)
