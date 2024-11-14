import dace
import numpy as np


N, d, Ti, Tj = (dace.symbol(s) for s in ('N', 'd', 'Ti', 'Tj'))


@dace.program
def custom_attention_dace(Q: dace.float32[N, d], K: dace.float32[N, d], V: dace.float32[N, d], O: dace.float32[N, d]):

    for ti in dace.map[0:N:Ti]:

        S = Q[ti:ti+Ti, :] @ np.transpose(K)
        m = np.max(S, axis=1)
        p_tilde = np.exp(S - m[:, np.newaxis])
        l = np.sum(p_tilde, axis=1)
        O[ti:ti+Ti, :] = (p_tilde @ V) / l[:, np.newaxis]
