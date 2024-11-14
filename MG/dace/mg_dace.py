import cupy
import dace
import numpy

from dace.transformation.auto.auto_optimize import auto_optimize, greedy_fuse, tile_wcrs
from dace.transformation.dataflow import MapReduceFusion, TaskletFusion, Vectorization


n0i, n0j, n0k = (dace.symbol(s, dtype=dace.int32) for s in ('n0i', 'n0j', 'n0k'))
n1i, n1j, n1k = (dace.symbol(s, dtype=dace.int64) for s in ('n1i', 'n1j', 'n1k'))


def compile_program_cpu(prog: dace.program, use_stencil_tiling: bool = True):
    sdfg = prog.to_sdfg(simplify=False)
    sdfg.simplify()
    auto_optimize(sdfg, dace.DeviceType.CPU, use_stencil_tiling=use_stencil_tiling)
    return sdfg.compile()


def compile_program_gpu(prog: dace.program, use_stencil_tiling: bool = True, use_wcr_tiling: bool = True):  
    sdfg = prog.to_sdfg(simplify=False)
    sdfg.simplify()
    auto_optimize(sdfg, dace.DeviceType.GPU, use_stencil_tiling=use_stencil_tiling, use_wcr_tiling=use_wcr_tiling, use_gpu_storage=True)
    return sdfg.compile()


@dace.program
def resid_dace(u: dace.float64[n0i, n0j, n0k], v: dace.float64[n0i, n0j, n0k], r: dace.float64[n0i, n0j, n0k], a: dace.float64[4]):

    for i, j, k in dace.map[1:u.shape[0] - 1, 1:u.shape[1] - 1, 1:u.shape[2] - 1]:

        r[i, j,
          k] = (v[i, j, k] - a[0] * u[i, j, k] - a[1] *
                (u[i - 1, j, k] + u[i + 1, j, k] + u[i, j - 1, k] + u[i, j + 1, k] + u[i, j, k - 1] + u[i, j, k + 1]) -
                a[2] * (u[i - 1, j - 1, k] + u[i - 1, j + 1, k] + u[i + 1, j - 1, k] + u[i + 1, j + 1, k] +
                        u[i - 1, j, k - 1] + u[i - 1, j, k + 1] + u[i + 1, j, k - 1] + u[i + 1, j, k + 1] +
                        u[i, j - 1, k - 1] + u[i, j - 1, k + 1] + u[i, j + 1, k - 1] + u[i, j + 1, k + 1]) - a[3] *
                (u[i - 1, j - 1, k - 1] + u[i - 1, j - 1, k + 1] + u[i - 1, j + 1, k - 1] + u[i - 1, j + 1, k + 1] +
                 u[i + 1, j - 1, k - 1] + u[i + 1, j - 1, k + 1] + u[i + 1, j + 1, k - 1] + u[i + 1, j + 1, k + 1]))

    # Periodic boundary conditions
    # axis = 1
    for i, j in dace.map[1:u.shape[0] - 1, 1:u.shape[1] - 1]:
        r[i, j, 0] = r[i, j, -2]
        r[i, j, -1] = r[i, j, 1]
    # axis = 2
    for i, k in dace.map[1:u.shape[0] - 1, 0:u.shape[2]]:
        r[i, 0, k] = r[i, -2, k]
        r[i, -1, k] = r[i, 1, k]
    # axis = 3
    for j, k in dace.map[0:u.shape[1], 0:u.shape[2]]:
        r[0, j, k] = r[-2, j, k]
        r[-1, j, k] = r[1, j, k]
    

@dace.program
def rprj3_dace(r: dace.float64[n0i, n0j, n0k], s: dace.float64[n1i, n1j, n1k]):

    for i, j, k in dace.map[1:s.shape[0] - 1, 1:s.shape[1] - 1, 1:s.shape[2] - 1]:

        s[i, j,
          k] = (0.5000 * r[2 * i, 2 * j, 2 * k] + 0.2500 *
                (r[2 * i - 1, 2 * j, 2 * k] + r[2 * i + 1, 2 * j, 2 * k] + r[2 * i, 2 * j - 1, 2 * k] +
                 r[2 * i, 2 * j + 1, 2 * k] + r[2 * i, 2 * j, 2 * k - 1] + r[2 * i, 2 * j, 2 * k + 1]) + 0.1250 *
                (r[2 * i - 1, 2 * j - 1, 2 * k] + r[2 * i - 1, 2 * j + 1, 2 * k] + r[2 * i + 1, 2 * j - 1, 2 * k] +
                 r[2 * i + 1, 2 * j + 1, 2 * k] + r[2 * i - 1, 2 * j, 2 * k - 1] + r[2 * i - 1, 2 * j, 2 * k + 1] +
                 r[2 * i + 1, 2 * j, 2 * k - 1] + r[2 * i + 1, 2 * j, 2 * k + 1] + r[2 * i, 2 * j - 1, 2 * k - 1] +
                 r[2 * i, 2 * j - 1, 2 * k + 1] + r[2 * i, 2 * j + 1, 2 * k - 1] + r[2 * i, 2 * j + 1, 2 * k + 1]) +
                0.0625 * (r[2 * i - 1, 2 * j - 1, 2 * k - 1] + r[2 * i - 1, 2 * j - 1, 2 * k + 1] +
                          r[2 * i - 1, 2 * j + 1, 2 * k - 1] + r[2 * i - 1, 2 * j + 1, 2 * k + 1] +
                          r[2 * i + 1, 2 * j - 1, 2 * k - 1] + r[2 * i + 1, 2 * j - 1, 2 * k + 1] +
                          r[2 * i + 1, 2 * j + 1, 2 * k - 1] + r[2 * i + 1, 2 * j + 1, 2 * k + 1]))

    # Periodic boundary conditions
    # axis = 1
    for i, j in dace.map[1:s.shape[0] - 1, 1:s.shape[1] - 1]:
        s[i, j, 0] = s[i, j, -2]
        s[i, j, -1] = s[i, j, 1]
    # axis = 2
    for i, k in dace.map[1:s.shape[0] - 1, 0:s.shape[2]]:
        s[i, 0, k] = s[i, -2, k]
        s[i, -1, k] = s[i, 1, k]
    # axis = 3
    for j, k in dace.map[0:s.shape[1], 0:s.shape[2]]:
        s[0, j, k] = s[-2, j, k]
        s[-1, j, k] = s[1, j, k]


@dace.program
def psinv_dace(r: dace.float64[n0i, n0j, n0k], u: dace.float64[n0i, n0j, n0k], c: dace.float64[4]):

    for i, j, k in dace.map[1:u.shape[0] - 1, 1:u.shape[1] - 1, 1:u.shape[2] - 1]:

        u[i, j,
          k] = (u[i, j, k] + c[0] * r[i, j, k] + c[1] *
                (r[i - 1, j, k] + r[i + 1, j, k] + r[i, j - 1, k] + r[i, j + 1, k] + r[i, j, k - 1] + r[i, j, k + 1]) +
                c[2] * (r[i - 1, j - 1, k] + r[i - 1, j + 1, k] + r[i + 1, j - 1, k] + r[i + 1, j + 1, k] +
                        r[i - 1, j, k - 1] + r[i - 1, j, k + 1] + r[i + 1, j, k - 1] + r[i + 1, j, k + 1] +
                        r[i, j - 1, k - 1] + r[i, j - 1, k + 1] + r[i, j + 1, k - 1] + r[i, j + 1, k + 1]) + c[3] *
                (r[i - 1, j - 1, k - 1] + r[i - 1, j - 1, k + 1] + r[i - 1, j + 1, k - 1] + r[i - 1, j + 1, k + 1] +
                 r[i + 1, j - 1, k - 1] + r[i + 1, j - 1, k + 1] + r[i + 1, j + 1, k - 1] + r[i + 1, j + 1, k + 1]))

    # Periodic boundary conditions
    # axis = 1
    for i, j in dace.map[1:u.shape[0] - 1, 1:u.shape[1] - 1]:
        u[i, j, 0] = u[i, j, -2]
        u[i, j, -1] = u[i, j, 1]
    # axis = 2
    for i, k in dace.map[1:u.shape[0] - 1, 0:u.shape[2]]:
        u[i, 0, k] = u[i, -2, k]
        u[i, -1, k] = u[i, 1, k]
    # axis = 3
    for j, k in dace.map[0:u.shape[1], 0:u.shape[2]]:
        u[0, j, k] = u[-2, j, k]
        u[-1, j, k] = u[1, j, k]


max_M = 256 + 2 + 1

@dace.program
def interp_dace(z: dace.float64[n0i, n0j, n0k], u: dace.float64[n1i, n1j, n1k]):

    # z1 = numpy.empty_like(z)
    # z2 = numpy.empty_like(z)
    # z3 = numpy.empty_like(z)
    z1 = numpy.empty((max_M, max_M, max_M), dtype=numpy.float64)
    z2 = numpy.empty((max_M, max_M, max_M), dtype=numpy.float64)
    z3 = numpy.empty((max_M, max_M, max_M), dtype=numpy.float64)

    for i3, i2, i1 in dace.map[0:z.shape[0] - 1, 0:z.shape[1] - 1, 0:z.shape[2]]:

        z1[i3, i2, i1] = z[i3, i2 + 1, i1] + z[i3, i2, i1]
        z2[i3, i2, i1] = z[i3 + 1, i2, i1] + z[i3, i2, i1]
        z3[i3, i2, i1] = z[i3 + 1, i2 + 1, i1] + z[i3 + 1, i2, i1] + z1[i3, i2, i1]

    for i3, i2, i1 in dace.map[0:z.shape[0] - 1, 0:z.shape[1] - 1, 0:z.shape[2] - 1]:

        u[2 * i3, 2 * i2, 2 * i1] = u[2 * i3, 2 * i2, 2 * i1] + z[i3, i2, i1]
        u[2 * i3, 2 * i2, 2 * i1 + 1] = u[2 * i3, 2 * i2, 2 * i1 + 1] + 0.5 * (z[i3, i2, i1 + 1] + z[i3, i2, i1])
        u[2 * i3, 2 * i2 + 1, 2 * i1] = u[2 * i3, 2 * i2 + 1, 2 * i1] + 0.5 * z1[i3, i2, i1]
        u[2 * i3, 2 * i2 + 1, 2 * i1 + 1] = u[2 * i3, 2 * i2 + 1, 2 * i1 + 1] + 0.25 * (z1[i3, i2, i1] + z1[i3, i2, i1 + 1])
        u[2 * i3 + 1, 2 * i2, 2 * i1] = u[2 * i3 + 1, 2 * i2, 2 * i1] + 0.5 * z2[i3, i2, i1]
        u[2 * i3 + 1, 2 * i2, 2 * i1 + 1] = u[2 * i3 + 1, 2 * i2, 2 * i1 + 1] + 0.25 * (z2[i3, i2, i1] + z2[i3, i2, i1 + 1])
        u[2 * i3 + 1, 2 * i2 + 1, 2 * i1] = u[2 * i3 + 1, 2 * i2 + 1, 2 * i1] + 0.25 * z3[i3, i2, i1]
        u[2 * i3 + 1, 2 * i2 + 1, 2 * i1 + 1] = u[2 * i3 + 1, 2 * i2 + 1, 2 * i1 + 1] + 0.125 * (z3[i3, i2, i1] + z3[i3, i2, i1 + 1])


@dace.program
def combo_dace(z: dace.float64[n0i, n0j, n0k], u: dace.float64[n1i, n1j, n1k], v: dace.float64[n1i, n1j, n1k], r: dace.float64[n1i, n1j, n1k],
               a: dace.float64[4], c: dace.float64[4]):
    interp_dace(z, u)
    resid_dace(u, v, r, a)
    psinv_dace(r, u, c)


@dace.program
def norm2u3_dace(r: dace.float64[n0i, n0j, n0k], dn: dace.int32):

    # s = 0.0
    # rnmu = 0.0
    # t0 = r[1:-1, 1:-1, 1:-1] * r[1:-1, 1:-1, 1:-1]
    # t1 = numpy.abs(r[1:-1, 1:-1, 1:-1])
    # dace.reduce(lambda a, b: a + b, t0, s)
    # dace.reduce(lambda a, b: max(a, b), t1, rnmu)
    # rnm2 = numpy.sqrt(s / dn)

    # s, rnmu = 0.0, 0.0
    # dace.reduce(lambda a, b: a + b, r[1:-1, 1:-1, 1:-1] * r[1:-1, 1:-1, 1:-1], s)
    # dace.reduce(lambda a, b: max(a, b), numpy.abs(r[1:-1, 1:-1, 1:-1]), rnmu)
    s = dace.reduce(lambda a, b: a + b, r[1:-1, 1:-1, 1:-1] * r[1:-1, 1:-1, 1:-1], identity=0.0)
    rnmu = dace.reduce(lambda a, b: max(a, b), numpy.abs(r[1:-1, 1:-1, 1:-1]), identity=0.0)
    rnm2 = numpy.sqrt(s / dn)

    return rnm2, rnmu


def get_mg_dace(framework: str):

    if framework == 'dace_cpu':
        print("Compiling DaCe versions")
        resid_func = compile_program_cpu(resid_dace)
        rprj3_func = compile_program_cpu(rprj3_dace)
        psinv_func = compile_program_cpu(psinv_dace)
        interp_func = compile_program_cpu(interp_dace, use_stencil_tiling=False)
        combo_func = compile_program_cpu(combo_dace, use_stencil_tiling=False)
        # sdfg = interp_dace.to_sdfg(simplify=True)  # Issue with persistent z1, z2, z3
        # interp_func = sdfg.compile()
        # sdfg = norm2u3_dace.to_sdfg(simplify=True)
        # auto_optimize(sdfg, dace.DeviceType.CPU)
        # sdfg.simplify()
        # auto_optimize(sdfg, dace.DeviceType.CPU)
        # norm2u3_func = compile_program_cpu(norm2u3_dace)
        sdfg = norm2u3_dace.to_sdfg(simplify=True)
        sdfg.apply_transformations_repeated([MapReduceFusion])
        tile_wcrs(sdfg, validate_all=True)
        greedy_fuse(sdfg, validate_all=True)
        norm2u3_func = sdfg.compile()  # Custom workflow to work around inability to tile two WCRs in a single Map
    elif framework == 'dace_gpu':
        print("Compiling DaCe versions")
        resid_func = compile_program_gpu(resid_dace)
        rprj3_func = compile_program_gpu(rprj3_dace)
        psinv_func = compile_program_gpu(psinv_dace)
        combo_func = compile_program_gpu(combo_dace, use_stencil_tiling=False)
        interp_func = compile_program_gpu(interp_dace, use_stencil_tiling=False)
        # sdfg = interp_dace.to_sdfg(simplify=True)
        # gpu_storage(sdfg)
        # sdfg.apply_gpu_transformations()
        # interp_func = sdfg.compile() # Issue with persistent z1, z2, z3
        norm2u3_func = compile_program_gpu(norm2u3_dace, use_wcr_tiling=False)
    
    return resid_func, rprj3_func, psinv_func, interp_func, combo_func, norm2u3_func


def mg3P_dace(lt, lb, m1, m2, m3, ir, u, v, r, a, c, n1, n2, n3, resid, rprj3, psinv, interp, combo=None):
    # --------------------------------------------------------------------
    # down cycle.
    # restrict the residual from the fine grid to the coarse
    # -------------------------------------------------------------------
    for k in range(lt, lb + 1 - 1, -1):
        j = k - 1
        szk = m1[k] * m2[k] * m3[k]
        szj = m1[j] * m2[j] * m3[j]
        r_sub_k = numpy.reshape(r[ir[k]:ir[k] + szk], (m3[k], m2[k], m1[k]))
        r_sub_j = numpy.reshape(r[ir[j]:ir[j] + szj], (m3[j], m2[j], m1[j]))
        rprj3(r=r_sub_k, s=r_sub_j, n0i=m3[k], n0j=m2[k], n0k=m1[k], n1i=m3[j], n1j=m3[j], n1k=m3[j])

    k2 = lb
    # --------------------------------------------------------------------
    # compute an approximate solution on the coarsest grid
    # --------------------------------------------------------------------
    szk = m1[k2] * m2[k2] * m3[k2]
    r_sub_p = numpy.reshape(r[ir[k2]:ir[k2] + szk], (m3[k2], m2[k2], m1[k2]))
    u_sub_p = numpy.reshape(u[ir[k2]:ir[k2] + szk], (m3[k2], m2[k2], m1[k2]))
    u_sub_p[:] = 0.0
    cupy.cuda.runtime.deviceSynchronize()
    psinv(r=r_sub_p, u=u_sub_p, c=c, n0i=m3[k2], n0j=m2[k2], n0k=m1[k2])

    for k in range(lb + 1, lt - 1 + 1):
        j = k - 1
        szk = m1[k] * m2[k] * m3[k]
        szj = m1[j] * m2[j] * m3[j]
        u_sub_k = numpy.reshape(u[ir[k]:ir[k] + szk], (m3[k], m2[k], m1[k]))
        u_sub_j = numpy.reshape(u[ir[j]:ir[j] + szj], (m3[j], m2[j], m1[j]))
        r_sub_k = numpy.reshape(r[ir[k]:ir[k] + szk], (m3[k], m2[k], m1[k]))
        # --------------------------------------------------------------------
        # prolongate from level k-1  to k
        # -------------------------------------------------------------------
        u_sub_k[:] = 0.0
        cupy.cuda.runtime.deviceSynchronize()
        if combo is not None:
            combo(z=u_sub_j, u=u_sub_k, v=r_sub_k, r=r_sub_k, a=a, c=c,
                  n0i=m3[j], n0j=m2[j], n0k=m1[j], n1i=m3[k], n1j=m2[k], n1k=m1[k])
        else:
            interp(z=u_sub_j, u=u_sub_k, n0i=m3[j], n0j=m2[j], n0k=m1[j], n1i=m3[k], n1j=m2[k], n1k=m1[k])
            # --------------------------------------------------------------------
            # compute residual for level k
            # --------------------------------------------------------------------
            resid(u=u_sub_k, v=r_sub_k, r=r_sub_k, a=a, n0i=m3[k], n0j=m2[k], n0k=m1[k])
            # --------------------------------------------------------------------
            # apply smoother
            # --------------------------------------------------------------------
            psinv(r=r_sub_k, u=u_sub_k, c=c, n0i=m3[k], n0j=m2[k], n0k=m1[k])

    j = lt - 1
    k2 = lt
    szj = m1[j] * m2[j] * m3[j]
    u_sub_j = numpy.reshape(u[ir[j]:ir[j] + szj], (m3[j], m2[j], m1[j]))
    my_u = numpy.reshape(u[:n1 * n2 * n3], (n3, n2, n1))
    my_v = numpy.reshape(v[:n1 * n2 * n3], (n3, n2, n1))
    my_r = numpy.reshape(r[:n1 * n2 * n3], (n3, n2, n1))

    if combo is not None:
        combo(z=u_sub_j, u=my_u, v=my_v, r=my_r, a=a, c=c,
              n0i=m3[j], n0j=m2[j], n0k=m1[j], n1i=n3, n1j=n2, n1k=n1)
    else:
        interp(z=u_sub_j, u=my_u, n0i=m3[j], n0j=m2[j], n0k=m1[j], n1i=n3, n1j=n2, n1k=n1)
        resid(u=my_u, v=my_v, r=my_r, a=a, n0i=n3, n0j=n2, n0k=n1)
        psinv(r=my_r, u=my_u, c=c, n0i=n3, n0j=n2, n0k=n1)


def mg_dace_full(lt, lb, m1, m2, m3, ir, dn, nit, u, v, r, a, c, n1, n2, n3, resid_func, norm2u3_func, rprj3_func, psinv_func, interp_func, combo_func=None):

    # if timeron:
    #     c_timers.timer_start(T_RESID2)
    resid_func(u=u.reshape(n3, n2, n1), v=v.reshape(n3, n2, n1), r=r.reshape(n3, n2, n1), a=a, n0i=n3, n0j=n2, n0k=n1)
    # cupy.cuda.runtime.deviceSynchronize()

    # if timeron:
    #     c_timers.timer_stop(T_RESID2)
    rnm2, rnmu = norm2u3_func(r=r.reshape(n3, n2, n1), dn=dn, n0i=n3, n0j=n2, n0k=n1)

    for it in range(1, nit+1):
        # if it == 1 or it == nit or (it%5) == 0:
        #     print("  iter %3d" % (it))
        # if timeron:
        #     c_timers.timer_start(T_MG3P)
        mg3P_dace(lt, lb, m1, m2, m3, ir, u, v, r, a, c, n1, n2, n3, resid_func, rprj3_func, psinv_func, interp_func, combo_func)
        # if timeron:
        #     c_timers.timer_stop(T_MG3P)
        # if timeron:
        #     c_timers.timer_start(T_RESID2)
        resid_func(u=u.reshape(n3, n2, n1), v=v.reshape(n3, n2, n1), r=r.reshape(n3, n2, n1), a=a, n0i=n3, n0j=n2, n0k=n1)

        # if timeron:
        #     c_timers.timer_stop(T_RESID2)

    rnm2, rnmu = norm2u3_func(r=r.reshape(n3, n2, n1), dn=dn, n0i=n3, n0j=n2, n0k=n1)
