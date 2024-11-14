import dace
import numpy as np


N, d, Ti, Tj = (dace.symbol(s) for s in ('N', 'd', 'Ti', 'Tj'))


@dace.program
def flash_attention_dace_4(Q: dace.float32[N, d], K: dace.float32[N, d], V: dace.float32[N, d], O: dace.float32[N, d]):

    for ti in dace.map[0:N:Ti]:

        m = np.full([Ti], -np.inf, Q.dtype)
        l = np.zeros([Ti], Q.dtype)
        S = np.empty([Ti, Tj], Q.dtype)
        Oi = np.zeros([Ti, d], Q.dtype)

        Qi = Q[ti:ti+Ti, :]

        for tj in range(0, N, Tj):

            S[:] = Qi @ np.transpose(K[tj:tj+Tj, :])

            max_row = np.max(S, axis=1)
            m_new = np.maximum(m, max_row)
            p_tilde = np.exp(S - m_new[:, np.newaxis])
            sum_row = np.sum(p_tilde, axis=1)
            l_tmp = l * np.exp(m - m_new)
            l_new = l_tmp + sum_row
            Oi[:] = (Oi * l_tmp[:, np.newaxis] + p_tilde @ V[tj:tj+Tj, :]) / l_new[:, np.newaxis]
            m[:] = m_new
            l[:] = l_new
        
        O[ti:ti+Ti, :] = Oi
