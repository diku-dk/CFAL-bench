import dace
import numpy as np

from dace.transformation.dataflow import MapExpansion, MapFusion, TaskletFusion


eps = np.finfo(np.float64).eps
N, iterations, tb_sz = (dace.symbol(s) for s in ('N', 'iterations', 'tb_sz'))


@dace.program
def nbody_dace_gpu(pos_mass: dace.float64[N, 4] @ dace.StorageType.GPU_Global,
                   vel: dace.float64[N, 3] @ dace.StorageType.GPU_Global,
                   dt: dace.float64):
    
    for _ in range(iterations):

        for block_start in dace.map[0:N:tb_sz] @ dace.ScheduleType.GPU_Device:
            for tid in dace.map[0:tb_sz] @ dace.ScheduleType.GPU_ThreadBlock:
                pos_mass_shared = dace.define_local((tb_sz, 4), dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
                accel = dace.define_local([3], dtype=dace.float64, storage=dace.StorageType.Register)
                accel[:] = 0.0
                pos_mass_i = pos_mass[block_start + tid]
                for j in dace.map[0:N:tb_sz] @ dace.ScheduleType.Sequential:
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
                    vel[block_start + tid] += accel * dt

        pos_mass[:, 0:3] += vel * dt


def get_nbody_dace_gpu():

    sdfg = nbody_dace_gpu.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated([MapFusion, TaskletFusion])
    sdfg.simplify()
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and len(node.map.params) > 1:
                MapExpansion.apply_to(sdfg, map_entry=node)
    
    return sdfg
