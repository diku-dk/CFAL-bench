import argparse
import dace
import numpy as np
import numpy.typing as npt
from timeit import repeat


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

        diff = pos_mass[0:3, np.newaxis, :] - pos_mass[0:3, :, np.newaxis]
        dist_sq = np.sum(diff * diff, axis=0)
        inv_dist = 1.0 / np.sqrt(dist_sq + eps)
        inv_dist3 = inv_dist * inv_dist * inv_dist
        accel = np.sum(diff * pos_mass[3, np.newaxis, :] * inv_dist3[np.newaxis, :, :], axis=2)
        vel[:] = vel + dt * accel
        pos_mass[0:3] = pos_mass[0:3] + dt * vel

N = dace.symbol("N")
iterations = dace.symbol("iterations")

@dace.program
def nbody_dace_cpu(pos_mass: dace.float64[4, N],
                   vel: dace.float64[3, N],
                   dt: dace.float64):

    for _ in range(iterations):

        diff = pos_mass[0:3, np.newaxis, :] - pos_mass[0:3, :, np.newaxis]
        dist_sq = np.sum(diff * diff, axis=0)
        inv_dist = 1.0 / np.sqrt(dist_sq + eps)
        inv_dist3 = inv_dist * inv_dist * inv_dist
        accel = np.sum(diff * pos_mass[3, np.newaxis, :] * inv_dist3[np.newaxis, :, :], axis=2)
        vel[:] = vel + dt * accel
        pos_mass[0:3] = pos_mass[0:3] + dt * vel


def relerror(val, ref):
    return np.linalg.norm(val - ref) / np.linalg.norm(ref)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-N", type=int, default=50000)
    argparser.add_argument("-iterations", type=int, default=10)	
    args = vars(argparser.parse_args())

    num_particles = 1000
    num_iterations = 10

    rng = np.random.default_rng(0)
    pos_mass = rng.random((4, num_particles))
    vel = rng.random((3, num_particles))
    dt = 0.01

    pos_mass_dace = np.asarray(pos_mass)
    vel_dace = np.asarray(vel)

    nbody_cpython(pos_mass, vel, dt, num_iterations)

    sdfg = nbody_dace_cpu.to_sdfg(simplify=True)
    sdfg.expand_library_nodes()

    # Permute map dimensions
    map_entries = []
    for node, parent in sdfg.all_nodes_recursive():
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
    
    # Expand maps
    exclude = set()
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node not in exclude:
            entries = MapExpansion.apply_to(parent.parent, map_entry=node)
            exclude.update(entries)

    # Find reduce-related nested SDFGs
    nested_sdfgs = {}
    for sd in sdfg.all_sdfgs_recursive():
        if sd is sdfg:
            continue
        nsdfg_node = sd.parent_nsdfg_node
        state = sd.parent
        for edge in state.out_edges(nsdfg_node):
            nested_sdfgs[edge.data.data] = nsdfg_node

    # Hoist reduce initialization states in the top-level SDFG
    HoistState.apply_to(sdfg, nsdfg=nested_sdfgs['accel'], permissive=True)
    HoistState.apply_to(sdfg, nsdfg=nested_sdfgs['dist_sq'], permissive=True)
    
    # Fuse maps
    sdfg.simplify()
    sdfg.apply_transformations_repeated([MapFusion])
    greedy_fuse(sdfg, False)
    sdfg.simplify()
    greedy_fuse(sdfg, False)
    auto_optimize(sdfg, dace.DeviceType.CPU)

    # Set zero initialization for WCR
    for edge, parent in sdfg.all_edges_recursive():
        if hasattr(edge.data, 'wcr') and edge.data.wcr is not None:
            dst = sdutil.get_global_memlet_path_dst(parent.parent, parent, edge)
            if isinstance(dst, dace.nodes.AccessNode):
                dst.setzero = True

    # Tasklet clean-up
    sdfg.apply_transformations_repeated([TaskletFusion])

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

    func = sdfg.compile()

    def _func():
        func(pos_mass=pos_mass, vel=vel, dt=dt, N=num_particles, iterations=num_iterations)
    
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
    print(f"DaCe CPU runtime: mean {mean} s ({flops} Gflop/s), std {std * 100 / mean:.2f}%", flush=True)
