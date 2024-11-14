import dace
import numpy as np

from dace.transformation.auto import auto_optimize


def get_flash_attention_dace_gpu():

    N, d, Ti, Tj = (dace.symbol(s) for s in ('N', 'd', 'Ti', 'Tj'))

    @dace.program
    def custom_attention_dace(Q: dace.float32[N, d], K: dace.float32[N, d], V: dace.float32[N, d], O: dace.float32[N, d]):

        for ti in dace.map[0:N:Ti]:

            S = Q[ti:ti+Ti, :] @ np.transpose(K)
            m = np.max(S, axis=1)
            p_tilde = np.exp(S - m[:, np.newaxis])
            l = np.sum(p_tilde, axis=1)
            O[ti:ti+Ti, :] = (p_tilde @ V) / l[:, np.newaxis]
    
    sdfg = custom_attention_dace.to_sdfg(simplify=False)
    auto_optimize.apply_gpu_storage(sdfg)
    for sd in sdfg.all_sdfgs_recursive():
        if sd.parent_sdfg is not None and sd.parent_sdfg is sdfg:
            sd.simplify()
            auto_optimize.auto_optimize(sd, dace.DeviceType.GPU, use_gpu_storage=True)
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.schedule = dace.ScheduleType.Sequential
    sdfg.simplify()
    return sdfg.compile()
