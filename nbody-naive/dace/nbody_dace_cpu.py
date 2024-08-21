import argparse
import dace
import numpy as np
import numpy.typing as npt
from timeit import repeat


from dace.sdfg import utils as sdutil
from dace.transformation.auto.auto_optimize import auto_optimize, greedy_fuse
from dace.transformation.dataflow import MapExpansion, MapFusion
from dace.transformation.interstate import HoistState


eps = np.finfo(np.float64).eps


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
        
        inv_dist = 1.0 / np.sqrt(np.sum(dist**2, axis=2) + eps)**3

        # f = np.empty((num_particles, num_particles, 3), dtype=np.float64)
        # for i, j in dace.map[0:num_particles, 0:num_particles]:
        #     f[i, j] = dist[i, j] * mass[j] * inv_dist[i, j]
        f = dist * mass[np.newaxis, :, np.newaxis] * inv_dist[:, :, np.newaxis]
        
        f_red = np.sum(f, axis=1)

        vel += dt * f_red
        pos += dt * vel


def nbody_python_3(pos_mass: npt.NDArray[np.float64],
                   vel: npt.NDArray[np.float64],
                   dt: np.float64,
                   iterations: int):
    
    N = pos_mass.shape[1]
    # dist = np.empty((3, N, N), dtype=np.float64)

    pos_ref = pos_mass[:3].T.copy(order='C')
    mass = pos_mass[3].copy(order='C')
    vel_ref = vel.T.copy(order='C')

    for _ in range(iterations):

        dist = pos_mass[0:3, np.newaxis, :] - pos_mass[0:3, :, np.newaxis]
        # for i in range(3):
        #     dist[i] = pos_mass[i, np.newaxis, :] - pos_mass[i, :, np.newaxis]

        # dist_ref = pos_ref[np.newaxis, :] - pos_ref[:, np.newaxis]
        # assert np.allclose(dist.transpose(1, 2, 0), dist_ref)

        inv_dist = 1.0 / np.sqrt(np.sum(dist * dist, axis=0) + eps)
        inv_dist3 = inv_dist * inv_dist * inv_dist

        # inv_dist3_ref = 1.0 / np.sqrt(np.sum(dist_ref**2, axis=2) + eps)**3
        # assert np.allclose(inv_dist3, inv_dist3_ref)

        accel = np.sum(dist * pos_mass[3, np.newaxis, :] * inv_dist3[np.newaxis, :, :], axis=2)

        # f = dist_ref * mass[np.newaxis, :, np.newaxis] * inv_dist3_ref[:, :, np.newaxis]
        # accel_ref = np.sum(f, axis=1)
        # assert np.allclose(accel.T, accel_ref)

        vel_new = vel + dt * accel
        pos_mass[0:3] = pos_mass[0:3] + dt * vel_new
        vel[:] = vel_new

        # vel_ref += dt * accel_ref
        # pos_ref += + dt * vel_ref
        # assert np.allclose(pos_mass[:3].T, pos_ref)
        # assert np.allclose(vel.T, vel_ref)
    
    for it in range(iterations):

        # dist = np.empty((num_particles, num_particles, 3), dtype=np.float64)
        # for i, j in dace.map[0:num_particles, 0:num_particles]:
        #     dist[i, j] = pos[j] - pos[i]
        dist_ref =  pos_ref[np.newaxis, :] - pos_ref[:, np.newaxis]
        
        inv_dist_ref = 1.0 / np.sqrt(np.sum(dist_ref**2, axis=2) + eps)**3

        # f = np.empty((num_particles, num_particles, 3), dtype=np.float64)
        # for i, j in dace.map[0:num_particles, 0:num_particles]:
        #     f[i, j] = dist[i, j] * mass[j] * inv_dist[i, j]
        f = dist_ref * mass[np.newaxis, :, np.newaxis] * inv_dist_ref[:, :, np.newaxis]
        
        f_red = np.sum(f, axis=1)

        vel_ref += dt * f_red
        pos_ref += dt * vel_ref

    print(relerror(pos_mass[:3].T, pos_ref))
    print(relerror(vel.T, vel_ref))



N = dace.symbol("N")
iterations = dace.symbol("iterations")

@dace.program
def nbody_dace_3(pos_mass: dace.float64[4, N], vel: dace.float64[3, N], dt: dace.float64):

    for _ in range(iterations):

        diff = pos_mass[0:3, np.newaxis, :] - pos_mass[0:3, :, np.newaxis]
        dist_sq = np.sum(diff * diff, axis=0)
        inv_dist = 1.0 / np.sqrt(dist_sq + eps)
        inv_dist3 = inv_dist * inv_dist * inv_dist
        accel = np.sum(diff * pos_mass[3, np.newaxis, :] * inv_dist3[np.newaxis, :, :], axis=2)
        accel_dt = dt * accel
        pos_mass[0:3] = pos_mass[0:3] + dt * (accel_dt + vel)
        vel[:] = accel_dt + vel


@dace.program
def nbody_dace_2(pos_mass: dace.float64[4, N], vel: dace.float64[3, N], dt: dace.float64):

    accel = np.empty((3, N), dtype=np.float64)

    for _ in range(iterations):

        for i in dace.map[0:N]:
            tmp = dace.define_local([3], dtype=dace.float64, storage=dace.StorageType.Register)
            tmp[:] = 0.0
            posi = pos_mass[0:3, i]
            for j in dace.map[0:N] @ dace.ScheduleType.Sequential:
                posj = pos_mass[:, j]
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
                    inv_dist = 1.0 / dace.math.sqrt(dist_sq + eps)
                    inv_dist3 = inv_dist * inv_dist * inv_dist
                    tmp_out[0] = tmp_in[0] + dist0 * mass_in * inv_dist3
                    tmp_out[1] = tmp_in[1] + dist1 * mass_in * inv_dist3
                    tmp_out[2] = tmp_in[2] + dist2 * mass_in * inv_dist3
            accel[:, i] = tmp

        for i in dace.map[0:N]:
            with dace.tasklet:
                acc_in << accel[0:3, i]
                dt_in << dt
                pos_in << pos_mass[0:3, i]
                vel_in << vel[0:3, i]
                pos_out >> pos_mass[0:3, i]
                vel_out >> vel[0:3, i]
                tmp0 = vel_in[0] + dt_in * acc_in[0]
                tmp1 = vel_in[1] + dt_in * acc_in[1]
                tmp2 = vel_in[2] + dt_in * acc_in[2]
                pos_out[0] = pos_in[0] + dt_in * tmp0
                pos_out[1] = pos_in[1] + dt_in * tmp1
                pos_out[2] = pos_in[2] + dt_in * tmp2
                vel_out[0] = tmp0
                vel_out[1] = tmp1
                vel_out[2] = tmp2


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
                inv_dist_out = 1.0 / dace.math.sqrt(dist_sq + eps)**3

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


@dace.program
def nbody_dace_opt(pos: dace.float64[N, 3],
                   vel: dace.float64[N, 3],
                   mass: dace.float64[N],
                   dt: dace.float64):
    
    accel = np.empty((N, 3), dtype=np.float64)

    for _ in range(iterations):

        for i in dace.map[0:N]:
            tmp = dace.define_local([3], dtype=dace.float64, storage=dace.StorageType.Register)
            tmp[:] = 0.0
            posi = pos[i]
            for j in dace.map[0:N] @ dace.ScheduleType.Sequential:
                posj = pos[j]
                with dace.tasklet:
                    pos_i << posi[0:3]
                    pos_j << posj[0:3]
                    mass_in << mass[j]
                    tmp_in << tmp[0:3]
                    tmp_out >> tmp[0:3]
                    dist0 = pos_j[0] - pos_i[0]
                    dist1 = pos_j[1] - pos_i[1]
                    dist2 = pos_j[2] - pos_i[2]
                    dist_sq = dist0*dist0 + dist1*dist1 + dist2*dist2
                    inv_dist = 1.0 / dace.math.sqrt(dist_sq + eps)
                    inv_dist3 = inv_dist * inv_dist * inv_dist
                    tmp_out[0] = tmp_in[0] + dist0 * mass_in * inv_dist3
                    tmp_out[1] = tmp_in[1] + dist1 * mass_in * inv_dist3
                    tmp_out[2] = tmp_in[2] + dist2 * mass_in * inv_dist3
            accel[i] = tmp

        for i in dace.map[0:N]:
            with dace.tasklet:
                acc_in << accel[i, 0:3]
                dt_in << dt
                pos_in << pos[i, 0:3]
                vel_in << vel[i, 0:3]
                pos_out >> pos[i, 0:3]
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


@dace.program
def nbody_dace_opt2(pos_mass: dace.float64[N, 4],
                    vel: dace.float64[N, 3],
                    dt: dace.float64):
    
    accel = np.empty((N, 3), dtype=np.float64)

    for _ in range(iterations):

        for i in dace.map[0:N]:
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
                    inv_dist = 1.0 / dace.math.sqrt(dist_sq + eps)
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
    # pos_dace_opt = np.empty((num_particles, 4), dtype=np.float64)
    # pos_dace_opt[:, :3] = pos
    # pos_dace_opt[:, 3] = mass
    pos_dace_opt = pos.copy()
    vel_dace_opt = vel.copy()
    # sdfg_opt(pos_mass=pos_dace_opt, vel=vel_dace_opt, dt=dt, N=num_particles, iterations=num_iterations)
    sdfg_opt(pos=pos_dace_opt, vel=vel_dace_opt, mass=mass, dt=dt, N=num_particles, iterations=num_iterations)
    print("Position error (optimized):", relerror(pos_dace_opt[:, 0:3], pos_ref))
    print("Velocity error (optimized):", relerror(vel_dace_opt, vel_ref))

    pos_mass = np.empty((4, num_particles), dtype=np.float64)
    pos_mass[:3, :] = np.copy(np.transpose(pos), order='C')
    pos_mass[3, :] = mass
    vel_T = np.copy(np.transpose(vel), order='C')
    print(vel_T.shape)

    assert np.allclose(pos_mass[:3].T, pos)
    assert np.allclose(vel_T.T, vel)
    assert np.allclose(pos_mass[3], mass)

    pos_mass_python = pos_mass.copy()
    vel_python = vel_T.copy()
    nbody_python_3(pos_mass_python, vel_python, dt, num_iterations)
    print("Position error (Python):", relerror(pos_mass_python[:3, :].T, pos_ref))
    print("Velocity error (Python):", relerror(vel_python.T, vel_ref))

    sdfg_3 = nbody_dace_3.to_sdfg(simplify=True)
    sdfg_3.expand_library_nodes()
    map_entries = []
    for node, parent in sdfg_3.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            if len(node.map.params) == 3:
                map_entries.append((node, parent, parent.parent))
                permutation = []
                for i, srng in enumerate(node.map.range.ranges):
                    if srng == (0, 2, 1):
                        permutation = [j for j in range(3) if j != i]
                        permutation.append(i)
                        break
                assert len(permutation) == 3
                node.map.params = [node.map.params[i] for i in permutation]
                node.map.range.ranges = [node.map.range.ranges[i] for i in permutation]
            elif len(node.map.params) == 2 and node.map.range.ranges[0] == (0, 2, 1):
                map_entries.append((node, parent, parent.parent))
                node.map.params = [node.map.params[i] for i in [1, 0]]
                node.map.range.ranges = [node.map.range.ranges[i] for i in [1, 0]]
    
    exclude = set()
    for node, parent in sdfg_3.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node not in exclude:
            # entries = MapExpansion.apply_to(parent.parent, map_entry=node, options={'expansion_limit': 1})
            entries = MapExpansion.apply_to(parent.parent, map_entry=node)
            exclude.update(entries)

    nested_sdfgs = {}
    for sd in sdfg_3.all_sdfgs_recursive():
        if sd is sdfg_3:
            continue
        nsdfg_node = sd.parent_nsdfg_node
        state = sd.parent
        for edge in state.out_edges(nsdfg_node):
            nested_sdfgs[edge.data.data] = nsdfg_node
    print(nested_sdfgs)

    HoistState.apply_to(sdfg_3, nsdfg=nested_sdfgs['accel'], permissive=True)
    HoistState.apply_to(sdfg_3, nsdfg=nested_sdfgs['dist_sq'], permissive=True)
    sdfg_3.simplify()
    sdfg_3.apply_transformations_repeated([MapFusion])
    greedy_fuse(sdfg_3, False)
    sdfg_3.simplify()
    greedy_fuse(sdfg_3, False)
    auto_optimize(sdfg_3, dace.DeviceType.CPU)
    for edge, parent in sdfg_3.all_edges_recursive():
        if hasattr(edge.data, 'wcr') and edge.data.wcr is not None:
            dst = sdutil.get_global_memlet_path_dst(parent.parent, parent, edge)
            if isinstance(dst, dace.nodes.AccessNode):
                dst.setzero = True
    pos_mass_dace_3 = pos_mass.copy()
    vel_dace_3 = vel_T.copy()
    sdfg_3(pos_mass=pos_mass_dace_3, vel=vel_dace_3, dt=dt, N=num_particles, iterations=num_iterations)
    print("Position error (version 3):", relerror(pos_mass_dace_3[:3, :].T, pos_ref))
    print("Velocity error (version 3):", relerror(np.transpose(vel_dace_3), vel_ref))

    sdfg_opt_2 = nbody_dace_2.to_sdfg(simplify=True)
    pos_mass_opt_2 = pos_mass.copy()
    vel_dace_opt_2 = vel_T.copy()    
    sdfg_opt_2(pos_mass=pos_mass_opt_2, vel=vel_dace_opt_2, dt=dt, N=num_particles, iterations=num_iterations)
    print("Position error (optimized 2):", relerror(pos_mass_opt_2[:3, :].T, pos_ref))
    print("Velocity error (optimized 2):", relerror(np.transpose(vel_dace_opt_2), vel_ref))

    # exit(0)

    # Benchmark
    num_particles = 50000
    num_iterations = 10
    pos = rng.random((num_particles, 3))
    # pos_mass = rng.random((num_particles, 4))
    vel = rng.random((num_particles, 3))
    mass = rng.random((num_particles, ))
    dt = 0.01

    func = sdfg_opt.compile()
    
    runtimes = repeat(lambda: func(pos=pos, vel=vel, mass=mass, dt=dt, N=num_particles, iterations=num_iterations),
                      number=1, repeat=10)
    median_time = np.median(runtimes)
    min_time = np.min(runtimes)
    flops = (18.0 * num_particles * num_particles + 12.0 * num_particles) * num_iterations / (1e9 * median_time)
    print(f"Runtime: median {median_time} s ({flops} Gflop/s), min {min_time} s", flush=True)
    print()

    pos_mass = rng.random((4, num_particles))
    vel = rng.random((3, num_particles))
    func = sdfg_opt_2.compile()

    runtimes = repeat(lambda: func(pos_mass=pos_mass, vel=vel, dt=dt, N=num_particles, iterations=num_iterations),
                      number=1, repeat=10)
    median_time = np.median(runtimes)
    min_time = np.min(runtimes)
    flops = (18.0 * num_particles * num_particles + 12.0 * num_particles) * num_iterations / (1e9 * median_time)
    print(f"Runtime (optimized 2): median {median_time} s ({flops} Gflop/s), min {min_time} s", flush=True)
    print()

    pos_mass = rng.random((4, num_particles))
    vel = rng.random((3, num_particles))
    func = sdfg_3.compile()

    runtimes = repeat(lambda: func(pos_mass=pos_mass, vel=vel, dt=dt, N=num_particles, iterations=num_iterations),
                      number=1, repeat=10)
    median_time = np.median(runtimes)
    min_time = np.min(runtimes)
    flops = (18.0 * num_particles * num_particles + 12.0 * num_particles) * num_iterations / (1e9 * median_time)
    print(f"Runtime (version 3): median {median_time} s ({flops} Gflop/s), min {min_time} s", flush=True)
    print()


