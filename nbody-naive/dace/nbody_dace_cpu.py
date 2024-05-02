import dace
import numpy as np
import numpy.typing as npt
from timeit import repeat


def nbody_python(pos: npt.NDArray[np.float64],
                 vel: npt.NDArray[np.float64],
                 mass: npt.NDArray[np.float64],
                 dt: np.float64,
                 iterations: int):
    
    num_particles = len(pos)

    for it in range(iterations):

        # dist = np.empty((num_particles, num_particles, 3), dtype=np.float64)
        # for i, j in dace.map[0:num_particles, 0:num_particles]:
        #     dist[i, j] = pos[j] - pos[i]
        dist =  pos[np.newaxis, :] - pos[:, np.newaxis]
        
        inv_dist = 1.0 / np.sqrt(np.sum(dist**2, axis=2) + 1e-9)**3

        # f = np.empty((num_particles, num_particles, 3), dtype=np.float64)
        # for i, j in dace.map[0:num_particles, 0:num_particles]:
        #     f[i, j] = dist[i, j] * mass[j] * inv_dist[i, j]
        f = dist * mass[np.newaxis, :, np.newaxis] * inv_dist[:, :, np.newaxis]
        
        f_red = np.sum(f, axis=1)

        vel += dt * f_red
        pos += dt * vel


N = dace.symbol("N")
iterations = dace.symbol("iterations")


@dace.program
def nbody_dace(pos: dace.float64[N, 3],
               vel: dace.float64[N, 3],
               mass: dace.float64[N],
               dt: dace.float64):


    dist = dace.ndarray([N, N, 3], dtype=dace.float64)
    inv_dist = dace.ndarray([N, N], dtype=dace.float64)
    f = dace.ndarray([N, N, 3], dtype=dace.float64)
    f_red = dace.ndarray([N, 3], dtype=dace.float64)

    for it in range(iterations):

        for i, j in dace.map[0:N, 0:N]:
            with dace.tasklet:
                pos_i << pos[i, 0:3]
                pos_j << pos[j, 0:3]
                dist_out >> dist[i, j, 0:3]
                dist_out[0] = pos_j[0] - pos_i[0]
                dist_out[1] = pos_j[1] - pos_i[1]
                dist_out[2] = pos_j[2] - pos_i[2]
        
        for i, j in dace.map[0:N, 0:N]:
            with dace.tasklet:
                dist_in << dist[i, j, 0:3]
                inv_dist_out >> inv_dist[i, j]
                dist_sq = dist_in[0]**2 + dist_in[1]**2 + dist_in[2]**2
                inv_dist_out = 1.0 / dace.math.sqrt(dist_sq + 1e-9)**3

        for i, j in dace.map[0:N, 0:N]:
            with dace.tasklet:
                dist_in << dist[i, j, 0:3]
                inv_dist_in << inv_dist[i, j]
                mass_in << mass[j]
                f_out >> f[i, j, 0:3]
                f_out[0] = dist_in[0] * mass_in * inv_dist_in
                f_out[1] = dist_in[1] * mass_in * inv_dist_in
                f_out[2] = dist_in[2] * mass_in * inv_dist_in
        
        dace.reduce(lambda a, b: a + b, f, f_red, axis=1, identity=0.0)
        
        for i in dace.map[0:N]:
            with dace.tasklet:
                f_red_in << f_red[i, 0:3]
                dt_in << dt
                pos_in << pos[i, 0:3]
                vel_in << vel[i, 0:3]
                pos_out >> pos[i, 0:3]
                vel_out >> vel[i, 0:3]
                tmp0 = vel_in[0] + dt_in * f_red_in[0]
                tmp1 = vel_in[1] + dt_in * f_red_in[1]
                tmp2 = vel_in[2] + dt_in * f_red_in[2]
                pos_out[0] = pos_in[0] + dt_in * tmp0
                pos_out[1] = pos_in[1] + dt_in * tmp1
                pos_out[2] = pos_in[2] + dt_in * tmp2
                vel_out[0] = tmp0
                vel_out[1] = tmp1
                vel_out[2] = tmp2


# @dace.program
# def nbody_dace_opt(pos: dace.float64[N, 3],
#                    vel: dace.float64[N, 3],
#                    mass: dace.float64[N],
#                    dt: dace.float64):
    
#     accel = np.empty((N, 3), dtype=np.float64)

#     for it in range(iterations):

#         for i in dace.map[0:N]:
#             tmp = np.zeros((3, ), dtype=np.float64)
#             for j in range(N):
#                 posi = pos[i, 0:3]
#                 posj = pos[j, 0:3]
#                 with dace.tasklet:
#                     pos_i << posi[0:3]
#                     pos_j << posj[0:3]
#                     mass_in << mass[j]
#                     tmp_in << tmp[0:3]
#                     tmp_out >> tmp[0:3]
#                     dist0 = pos_j[0] - pos_i[0]
#                     dist1 = pos_j[1] - pos_i[1]
#                     dist2 = pos_j[2] - pos_i[2]
#                     # dist_sq = dist0**2 + dist1**2 + dist2**2
#                     # inv_dist = 1.0 / dace.math.sqrt(dist_sq + 1e-9)**3
#                     # tmp_out[0] = tmp_in[0] + dist0 * mass_in * inv_dist
#                     # tmp_out[1] = tmp_in[1] + dist1 * mass_in * inv_dist
#                     # tmp_out[2] = tmp_in[2] + dist2 * mass_in * inv_dist
#                     dist_sq = dist0*dist0 + dist1*dist1 + dist2*dist2
#                     inv_dist = 1.0 / dace.math.sqrt(dist_sq + 1e-9)
#                     inv_dist3 = inv_dist * inv_dist * inv_dist
#                     tmp_out[0] = tmp_in[0] + dist0 * mass_in * inv_dist3
#                     tmp_out[1] = tmp_in[1] + dist1 * mass_in * inv_dist3
#                     tmp_out[2] = tmp_in[2] + dist2 * mass_in * inv_dist3
#             accel[i, 0:3] = tmp

#         for i in dace.map[0:N]:
#             with dace.tasklet:
#                 acc_in << accel[i, 0:3]
#                 dt_in << dt
#                 pos_in << pos[i, 0:3]
#                 vel_in << vel[i, 0:3]
#                 pos_out >> pos[i, 0:3]
#                 vel_out >> vel[i, 0:3]
#                 tmp0 = vel_in[0] + dt_in * acc_in[0]
#                 tmp1 = vel_in[1] + dt_in * acc_in[1]
#                 tmp2 = vel_in[2] + dt_in * acc_in[2]
#                 pos_out[0] = pos_in[0] + dt_in * tmp0
#                 pos_out[1] = pos_in[1] + dt_in * tmp1
#                 pos_out[2] = pos_in[2] + dt_in * tmp2
#                 vel_out[0] = tmp0
#                 vel_out[1] = tmp1
#                 vel_out[2] = tmp2

@dace.program
def nbody_dace_opt(pos_mass: dace.float64[N, 4],
                   vel: dace.float64[N, 3],
                   dt: dace.float64):
    
    accel = np.empty((N, 3), dtype=np.float64)

    for _ in range(iterations):

        for i in dace.map[0:N]:
            # tmp = np.zeros((3, ), dtype=np.float64)
            tmp = dace.define_local([3], dtype=dace.float64, storage=dace.StorageType.Register)
            tmp[:] = 0.0
            posi = pos_mass[i]
            for j in dace.map[0:N] @ dace.ScheduleType.Sequential:
                posj = pos_mass[j]
                with dace.tasklet:
                    pos_i << posi[0:3]
                    pos_j << posj[0:3]
                    mass_in << posj[3]
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
            accel[i] = tmp

        for i in dace.map[0:N]:
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


def relerror(val, ref):
    return np.linalg.norm(val - ref) / np.linalg.norm(ref)


if __name__ == "__main__":

    # Initialize data
    num_particles = 100
    num_iterations = 10
    rng = np.random.default_rng(0)
    pos = rng.random((num_particles, 3))
    vel = rng.random((num_particles, 3))
    mass = rng.random((num_particles, ))
    dt = 0.01

    # Run reference implementation
    pos_ref = pos.copy()
    vel_ref = vel.copy()
    nbody_python(pos_ref, vel_ref, mass, dt, num_iterations)

    # Run reference DaCe implementation
    sdfg = nbody_dace.to_sdfg(simplify=True)
    pos_dace = pos.copy()
    vel_dace = vel.copy()
    sdfg(pos=pos_dace, vel=vel_dace, mass=mass, dt=dt, N=num_particles, iterations=num_iterations)
    print("Position error:", relerror(pos_dace, pos_ref))
    print("Velocity error:", relerror(vel_dace, vel_ref))

    # Run optimized DaCe implementation
    sdfg_opt = nbody_dace_opt.to_sdfg(simplify=True)
    pos_dace_opt = np.empty((num_particles, 4), dtype=np.float64)
    pos_dace_opt[:, :3] = pos
    pos_dace_opt[:, 3] = mass
    # pos_dace_opt = pos.copy()
    vel_dace_opt = vel.copy()
    sdfg_opt(pos_mass=pos_dace_opt, vel=vel_dace_opt, dt=dt, N=num_particles, iterations=num_iterations)
    print("Position error (optimized):", relerror(pos_dace_opt[:, 0:3], pos_ref))
    print("Velocity error (optimized):", relerror(vel_dace_opt, vel_ref))

    # Benchmark
    num_particles = 10000
    num_iterations = 10
    # pos = rng.random((num_particles, 3))
    pos_mass = rng.random((num_particles, 4))
    vel = rng.random((num_particles, 3))
    # mass = rng.random((num_particles, ))
    dt = 0.01

    func = sdfg_opt.compile()
    runtimes = repeat(lambda: func(pos_mass=pos_mass, vel=vel, dt=dt, N=num_particles, iterations=num_iterations),
                      number=1, repeat=20)
    median_time = np.median(runtimes)
    min_time = np.min(runtimes)
    flops = (18.0 * num_particles * num_particles + 12.0 * num_particles) * num_iterations / (1e9 * median_time)
    print(f"Runtime: median {median_time} s ({flops} Gflop/s), min {min_time} s", flush=True)
    print()


