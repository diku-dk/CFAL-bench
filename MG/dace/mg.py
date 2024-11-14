# ------------------------------------------------------------------------------
#
# The original NPB 3.4.1 version was written in Fortran and belongs to:
# 	http://www.nas.nasa.gov/Software/NPB/
#
# Authors of the Fortran code:
#	E. Barszcz
#	P. Frederickson
#	A. Woo
#	M. Yarrow
#
# ------------------------------------------------------------------------------
#
# The serial C++ version is a translation of the original NPB 3.4.1
# Serial C++ version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER
#
# Authors of the C++ code:
# 	Dalvan Griebler <dalvangriebler@gmail.com>
# 	Gabriell Araujo <hexenoften@gmail.com>
# 	Júnior Löff <loffjh@gmail.com>
#
# ------------------------------------------------------------------------------
#
# The serial Python version is a translation of the NPB serial C++ version
# Serial Python version: https://github.com/danidomenico/NPB-PYTHON/tree/master/NPB-SER
#
# Authors of the Python code:
#	LUPS (Laboratory of Ubiquitous and Parallel Systems)
#	UFPEL (Federal University of Pelotas)
#	Pelotas, Rio Grande do Sul, Brazil
#
# ------------------------------------------------------------------------------

import argparse
import sys
import os
import numpy
import math


# from numba import njit
def njit(f):
    return f


# DaCe imports and symbols
import cupy
import dace
from dace.transformation.auto.auto_optimize import auto_optimize, greedy_fuse, tile_wcrs
from dace.transformation.dataflow import MapReduceFusion, TaskletFusion, Vectorization

from mg_dace import get_mg_dace, mg_dace_full

n0i, n0j, n0k = (dace.symbol(s, dtype=dace.int32) for s in ('n0i', 'n0j', 'n0k'))
n1i, n1j, n1k = (dace.symbol(s, dtype=dace.int64) for s in ('n1i', 'n1j', 'n1k'))

# Local imports
sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common')))
from common import npbparams
from common.c_randdp import vranlc
from common.c_randdp import randlc
from common import c_timers
from common import c_print_results

# --------------------------------------------------------------------
# this is the serial version of the app benchmark 1,
# the "embarassingly parallel" benchmark.
# --------------------------------------------------------------------
# M is the Log_2 of the number of complex pairs of uniform (0, 1) random
# numbers. MK is the Log_2 of the size of each batch of uniform random
# numbers.  MK can be set for convenience on a given system, since it does
# not affect the results.
# --------------------------------------------------------------------

# Global variables
NM = 0  # actual dimension including ghost cells for communications
NV = 0  # size of rhs array
NR = 0  # size of residual array
MAXLEVEL = 0  # maximum number of levels
M = 0  # set at m=1024, can handle cases up to 1024^3 case
MM = 10
A = pow(5.0, 13.0)
X = 314159265.0
T_INIT = 0
T_BENCH = 1
T_MG3P = 2
T_PSINV = 3
T_RESID = 4
T_RESID2 = 5
T_RPRJ3 = 6
T_INTERP = 7
T_NORM2 = 8
T_COMM3 = 9
T_LAST = 10

nx = None
ny = None
nz = None
m1 = None
m2 = None
m3 = None
ir = None
debug_vec = numpy.repeat(npbparams.DEBUG_DEFAULT, 8)
u = None
v = None
r = None

is1 = 0
is2 = 0
is3 = 0
ie1 = 0
ie2 = 0
ie3 = 0
lt = 0
lb = 0

timeron = False


def set_global_variables():
    global NM, NV, NR, MAXLEVEL, M
    global nx, ny, nz, m1, m2, m3, ir
    global u, v, r

    NM = 2 + (1 << npbparams.LM)
    NV = npbparams.ONE * (2 + (1 << npbparams.NDIM1)) * (2 + (1 << npbparams.NDIM2)) * (2 + (1 << npbparams.NDIM3))
    NR = int((NV + NM * NM + 5 * NM + 7 * npbparams.LM + 6) / 7) * 8
    MAXLEVEL = npbparams.LT_DEFAULT + 1
    M = NM + 1

    nx = numpy.empty(MAXLEVEL + 1, dtype=numpy.int32)
    ny = numpy.empty(MAXLEVEL + 1, dtype=numpy.int32)
    nz = numpy.empty(MAXLEVEL + 1, dtype=numpy.int32)
    m1 = numpy.empty(MAXLEVEL + 1, dtype=numpy.int32)
    m2 = numpy.empty(MAXLEVEL + 1, dtype=numpy.int32)
    m3 = numpy.empty(MAXLEVEL + 1, dtype=numpy.int32)
    ir = numpy.empty(MAXLEVEL + 1, dtype=numpy.int32)

    u = numpy.empty(NR, dtype=numpy.float64)
    v = numpy.empty(NV, dtype=numpy.float64)
    r = numpy.empty(NR, dtype=numpy.float64)


#END set_global_variables()


# --------------------------------------------------------------------
# interp adds the trilinear interpolation of the correction
# from the coarser grid to the current approximation: u = u + Qu'
#
# observe that this  implementation costs  16A + 4M, where
# A and M denote the costs of addition and multiplication.
# note that this vectorizes, and is also fine for cache
# based machines. vector machines may get slightly better
# performance however, with 8 separate "do i1" loops, rather than 4.
# --------------------------------------------------------------------
#static void interp(void* pointer_z, int mm1, int mm2, int mm3, void* pointer_u, int n1, int n2, int n3, int k)
@njit
def interp(pointer_z, mm1, mm2, mm3, pointer_u, n1, n2, n3, k):
    #double (*z)[mm2][mm1] = (double (*)[mm2][mm1])pointer_z;
    #double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;
    z = pointer_z  #(i3*mm2+i2)*mm1 + i1
    u = pointer_u  #(i3*n2+i2)*n1 + i1

    # --------------------------------------------------------------------
    # note that m = 1037 in globals.h but for this only need to be
    # 535 to handle up to 1024^3
    # integer m
    # parameter( m=535 )
    # --------------------------------------------------------------------
    z1 = numpy.empty(shape=M, dtype=numpy.float64)
    z2 = numpy.empty(shape=M, dtype=numpy.float64)
    z3 = numpy.empty(shape=M, dtype=numpy.float64)

    #if timeron: Not supported by @njit
    #	c_timers.timer_start(T_INTERP)
    if n1 != 3 and n2 != 3 and n3 != 3:
        for i3 in range(mm3 - 1):
            for i2 in range(mm2 - 1):
                for i1 in range(mm1):
                    z1[i1] = z[(i3 * mm2 +
                                (i2 + 1)) * mm1 + i1] + z[(i3 * mm2 + i2) * mm1 + i1]  #z[i3][i2+1][i1] + z[i3][i2][i1]
                    z2[i1] = z[((i3 + 1) * mm2 + i2) * mm1 + i1] + z[(i3 * mm2 + i2) * mm1 +
                                                                     i1]  #z[i3+1][i2][i1] + z[i3][i2][i1]
                    z3[i1] = z[((i3 + 1) * mm2 + (i2 + 1)) * mm1 + i1] + z[(
                        (i3 + 1) * mm2 + i2) * mm1 + i1] + z1[i1]  #z[i3+1][i2+1][i1] + z[i3+1][i2][i1] + z1[i1]

                for i1 in range(mm1 - 1):
                    #u[2*i3][2*i2][2*i1] = u[2*i3][2*i2][2*i1] + z[i3][i2][i1]
                    u[((2 * i3) * n2 + (2 * i2)) * n1 + (2 * i1)] = u[((2 * i3) * n2 + (2 * i2)) * n1 +
                                                                      (2 * i1)] + z[(i3 * mm2 + i2) * mm1 + i1]
                    #u[2*i3][2*i2][2*i1+1] = u[2*i3][2*i2][2*i1+1] +0.5*(z[i3][i2][i1+1]+z[i3][i2][i1])
                    u[((2 * i3) * n2 + (2 * i2)) * n1 +
                      (2 * i1 + 1)] = u[((2 * i3) * n2 + (2 * i2)) * n1 +
                                        (2 * i1 + 1)] + 0.5 * (z[(i3 * mm2 + i2) * mm1 +
                                                                 (i1 + 1)] + z[(i3 * mm2 + i2) * mm1 + i1])

                for i1 in range(mm1 - 1):
                    #u[2*i3][2*i2+1][2*i1] = u[2*i3][2*i2+1][2*i1] +0.5 * z1[i1]
                    u[((2 * i3) * n2 + (2 * i2 + 1)) * n1 + (2 * i1)] = u[((2 * i3) * n2 + (2 * i2 + 1)) * n1 +
                                                                          (2 * i1)] + 0.5 * z1[i1]
                    #u[2*i3][2*i2+1][2*i1+1] = u[2*i3][2*i2+1][2*i1+1] +0.25*( z1[i1] + z1[i1+1] )
                    u[((2 * i3) * n2 + (2 * i2 + 1)) * n1 +
                      (2 * i1 + 1)] = u[((2 * i3) * n2 + (2 * i2 + 1)) * n1 +
                                        (2 * i1 + 1)] + 0.25 * (z1[i1] + z1[i1 + 1])

                for i1 in range(mm1 - 1):
                    #u[2*i3+1][2*i2][2*i1] = u[2*i3+1][2*i2][2*i1] +0.5 * z2[i1]
                    u[((2 * i3 + 1) * n2 + (2 * i2)) * n1 + (2 * i1)] = u[((2 * i3 + 1) * n2 + (2 * i2)) * n1 +
                                                                          (2 * i1)] + 0.5 * z2[i1]
                    #u[2*i3+1][2*i2][2*i1+1] = u[2*i3+1][2*i2][2*i1+1] +0.25*( z2[i1] + z2[i1+1] )
                    u[((2 * i3 + 1) * n2 + (2 * i2)) * n1 +
                      (2 * i1 + 1)] = u[((2 * i3 + 1) * n2 + (2 * i2)) * n1 +
                                        (2 * i1 + 1)] + 0.25 * (z2[i1] + z2[i1 + 1])

                for i1 in range(mm1 - 1):
                    #u[2*i3+1][2*i2+1][2*i1] = u[2*i3+1][2*i2+1][2*i1] +0.25* z3[i1]
                    u[((2 * i3 + 1) * n2 + (2 * i2 + 1)) * n1 + (2 * i1)] = u[((2 * i3 + 1) * n2 + (2 * i2 + 1)) * n1 +
                                                                              (2 * i1)] + 0.25 * z3[i1]
                    #u[2*i3+1][2*i2+1][2*i1+1] = u[2*i3+1][2*i2+1][2*i1+1] +0.125*( z3[i1] + z3[i1+1] )
                    u[((2 * i3 + 1) * n2 + (2 * i2 + 1)) * n1 +
                      (2 * i1 + 1)] = u[((2 * i3 + 1) * n2 + (2 * i2 + 1)) * n1 +
                                        (2 * i1 + 1)] + 0.125 * (z3[i1] + z3[i1 + 1])
            #END for i2 in range(mm2-1):
        #END for i3 in range(mm3-1):
    else:
        if n1 == 3:
            d1 = 2
            t1 = 1
        else:
            d1 = 1
            t1 = 0

        if n2 == 3:
            d2 = 2
            t2 = 1
        else:
            d2 = 1
            t2 = 0

        if n3 == 3:
            d3 = 2
            t3 = 1
        else:
            d3 = 1
            t3 = 0

        for i3 in range(d3, mm3 - 1 + 1):
            for i2 in range(d2, mm2 - 1 + 1):
                for i1 in range(d1, mm1 - 1 + 1):
                    #u[2*i3-d3-1][2*i2-d2-1][2*i1-d1-1] = u[2*i3-d3-1][2*i2-d2-1][2*i1-d1-1] +z[i3-1][i2-1][i1-1]
                    u[((2 * i3 - d3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 +
                      (2 * i1 - d1 - 1)] = u[((2 * i3 - d3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 +
                                             (2 * i1 - d1 - 1)] + z[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]

                for i1 in range(1, mm1 - 1 + 1):
                    #u[2*i3-d3-1][2*i2-d2-1][2*i1-t1-1] = u[2*i3-d3-1][2*i2-d2-1][2*i1-t1-1] +0.5*(z[i3-1][i2-1][i1]+z[i3-1][i2-1][i1-1])
                    u[((2 * i3 - d3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 +
                      (2 * i1 - t1 - 1)] = (u[((2 * i3 - d3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - t1 - 1)] +
                                            0.5 * (z[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + i1] + z[((i3 - 1) * mm2 +
                                                                                                  (i2 - 1)) * mm1 +
                                                                                                 (i1 - 1)]))

            for i2 in range(1, mm2 - 1 + 1):
                for i1 in range(d1, mm1 - 1 + 1):
                    #u[2*i3-d3-1][2*i2-t2-1][2*i1-d1-1] = u[2*i3-d3-1][2*i2-t2-1][2*i1-d1-1] +0.5*(z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1])
                    u[((2 * i3 - d3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 +
                      (2 * i1 - d1 - 1)] = (u[((2 * i3 - d3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - d1 - 1)] +
                                            0.5 * (z[((i3 - 1) * mm2 + i2) * mm1 +
                                                     (i1 - 1)] + z[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]))

                for i1 in range(1, mm1 - 1 + 1):
                    #u[2*i3-d3-1][2*i2-t2-1][2*i1-t1-1] = u[2*i3-d3-1][2*i2-t2-1][2*i1-t1-1] +
                    #0.25*(z[i3-1][i2][i1]+z[i3-1][i2-1][i1] +z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1])
                    u[((2 * i3 - d3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 +
                      (2 * i1 - t1 - 1)] = (u[((2 * i3 - d3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - t1 - 1)] +
                                            0.25 * (z[((i3 - 1) * mm2 + i2) * mm1 + i1] + z[(
                                                (i3 - 1) * mm2 +
                                                (i2 - 1)) * mm1 + i1] + z[((i3 - 1) * mm2 + i2) * mm1 +
                                                                          (i1 - 1)] + z[((i3 - 1) * mm2 +
                                                                                         (i2 - 1)) * mm1 + (i1 - 1)]))
        #END for i3 in range(d3, mm3-1+1):

        for i3 in range(1, mm3 - 1 + 1):
            for i2 in range(d2, mm2 - 1 + 1):
                for i1 in range(d1, mm1 - 1 + 1):
                    #u[2*i3-t3-1][2*i2-d2-1][2*i1-d1-1] = u[2*i3-t3-1][2*i2-d2-1][2*i1-d1-1] + 0.5*(z[i3][i2-1][i1-1]+z[i3-1][i2-1][i1-1])
                    u[((2 * i3 - t3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 +
                      (2 * i1 - d1 - 1)] = (u[((2 * i3 - t3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - d1 - 1)] +
                                            0.5 * (z[(i3 * mm2 + (i2 - 1)) * mm1 +
                                                     (i1 - 1)] + z[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]))

                for i1 in range(1, mm1 - 1 + 1):
                    #u[2*i3-t3-1][2*i2-d2-1][2*i1-t1-1] = u[2*i3-t3-1][2*i2-d2-1][2*i1-t1-1] +
                    #		0.25*(z[i3][i2-1][i1]+z[i3][i2-1][i1-1] +z[i3-1][i2-1][i1]+z[i3-1][i2-1][i1-1])
                    u[((2 * i3 - t3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - t1 - 1)] = (
                        u[((2 * i3 - t3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - t1 - 1)] + 0.25 *
                        (z[(i3 * mm2 + (i2 - 1)) * mm1 + i1] + z[(i3 * mm2 + (i2 - 1)) * mm1 + (i1 - 1)] + z[(
                            (i3 - 1) * mm2 + (i2 - 1)) * mm1 + i1] + z[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]))

            for i2 in range(1, mm2 - 1 + 1):
                for i1 in range(d1, mm1 - 1 + 1):
                    #u[2*i3-t3-1][2*i2-t2-1][2*i1-d1-1] = u[2*i3-t3-1][2*i2-t2-1][2*i1-d1-1] +
                    #		0.25*(z[i3][i2][i1-1]+z[i3][i2-1][i1-1] +z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1])
                    u[((2 * i3 - t3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 +
                      (2 * i1 - d1 - 1)] = (u[((2 * i3 - t3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - d1 - 1)] +
                                            0.25 * (z[(i3 * mm2 + i2) * mm1 +
                                                      (i1 - 1)] + z[(i3 * mm2 + (i2 - 1)) * mm1 +
                                                                    (i1 - 1)] + z[((i3 - 1) * mm2 + i2) * mm1 +
                                                                                  (i1 - 1)] + z[((i3 - 1) * mm2 +
                                                                                                 (i2 - 1)) * mm1 +
                                                                                                (i1 - 1)]))

                for i1 in range(1, mm1 - 1 + 1):
                    #u[2*i3-t3-1][2*i2-t2-1][2*i1-t1-1] = u[2*i3-t3-1][2*i2-t2-1][2*i1-t1-1] +
                    #		0.125*(z[i3][i2][i1]+z[i3][i2-1][i1]
                    #			+z[i3][i2][i1-1]+z[i3][i2-1][i1-1]
                    #			+z[i3-1][i2][i1]+z[i3-1][i2-1][i1]
                    #			+z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1])
                    u[((2 * i3 - t3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - t1 - 1)] = (
                        u[((2 * i3 - t3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - t1 - 1)] + 0.125 *
                        (z[(i3 * n2 + i2) * n1 + i1] + z[(i3 * n2 + (i2 - 1)) * n1 + i1] + z[
                            (i3 * n2 + i2) * n1 +
                            (i1 - 1)] + z[(i3 * n2 + (i2 - 1)) * n1 +
                                          (i1 - 1)] + z[((i3 - 1) * n2 + i2) * n1 + i1] + z[(
                                              (i3 - 1) * n2 + (i2 - 1)) * n1 + i1] + z[((i3 - 1) * n2 + i2) * n1 +
                                                                                       (i1 - 1)] + z[((i3 - 1) * n2 +
                                                                                                      (i2 - 1)) * n1 +
                                                                                                     (i1 - 1)]))
        #END for i3 in range(1, mm3-1+1):
    #END if n1 != 3 and n2 != 3 and n3 != 3:
    #if timeron: Not supported by @njit
    #	c_timers.timer_stop(T_INTERP)

    if debug_vec[0] >= 1:
        rep_nrm(z, mm1, mm2, mm3, "z: inter", k - 1)
        rep_nrm(u, n1, n2, n3, "u: inter", k)

    if debug_vec[5] >= k:
        showall(z, mm1, mm2, mm3)
        showall(u, n1, n2, n3)


#END interp()


# @dace.program
# def interp_dace(z: dace.float64[n0i, n0j, n0k], u: dace.float64[n1i, n1j, n1k]):

#     z1 = numpy.empty_like(z)
#     z2 = numpy.empty_like(z)
#     z3 = numpy.empty_like(z)

#     for i3, i2 in dace.map[0:z.shape[0] - 1, 0:z.shape[1] - 1]:

#         for i1 in dace.map[0:z.shape[2]]:
#             z1[i3, i2, i1] = z[i3, i2 + 1, i1] + z[i3, i2, i1]
#             z2[i3, i2, i1] = z[i3 + 1, i2, i1] + z[i3, i2, i1]
#             z3[i3, i2, i1] = z[i3 + 1, i2 + 1, i1] + z[i3 + 1, i2, i1] + z1[i3, i2, i1]

#         for i1 in dace.map[0:z.shape[2] - 1]:
#             u[2 * i3, 2 * i2, 2 * i1] = u[2 * i3, 2 * i2, 2 * i1] + z[i3, i2, i1]
#             u[2 * i3, 2 * i2, 2 * i1 + 1] = u[2 * i3, 2 * i2, 2 * i1 + 1] + 0.5 * (z[i3, i2, i1 + 1] + z[i3, i2, i1])
#             u[2 * i3, 2 * i2 + 1, 2 * i1] = u[2 * i3, 2 * i2 + 1, 2 * i1] + 0.5 * z1[i3, i2, i1]
#             u[2 * i3, 2 * i2 + 1,
#               2 * i1 + 1] = u[2 * i3, 2 * i2 + 1, 2 * i1 + 1] + 0.25 * (z1[i3, i2, i1] + z1[i3, i2, i1 + 1])
#             u[2 * i3 + 1, 2 * i2, 2 * i1] = u[2 * i3 + 1, 2 * i2, 2 * i1] + 0.5 * z2[i3, i2, i1]
#             u[2 * i3 + 1, 2 * i2,
#               2 * i1 + 1] = u[2 * i3 + 1, 2 * i2, 2 * i1 + 1] + 0.25 * (z2[i3, i2, i1] + z2[i3, i2, i1 + 1])
#             u[2 * i3 + 1, 2 * i2 + 1, 2 * i1] = u[2 * i3 + 1, 2 * i2 + 1, 2 * i1] + 0.25 * z3[i3, i2, i1]
#             u[2 * i3 + 1, 2 * i2 + 1,
#               2 * i1 + 1] = u[2 * i3 + 1, 2 * i2 + 1, 2 * i1 + 1] + 0.125 * (z3[i3, i2, i1] + z3[i3, i2, i1 + 1])


# @dace.program
# def interp_dace(z: dace.float64[n0i, n0j, n0k], u: dace.float64[n1i, n1j, n1k]):

#     # z1 = numpy.empty_like(z)
#     # z2 = numpy.empty_like(z)
#     # z3 = numpy.empty_like(z)

#     for i3, i2 in dace.map[0:z.shape[0] - 1, 0:z.shape[1] - 1]:

#         z10 = z[i3, i2 + 1, 0] + z[i3, i2, 0]
#         z20 = z[i3 + 1, i2, 0] + z[i3, i2, 0]
#         z30 = z[i3 + 1, i2 + 1, 0] + z[i3 + 1, i2, 0] + z10

#         for i1 in range(z.shape[2] - 1):

#             z11 = z[i3, i2 + 1, i1 + 1] + z[i3, i2, i1 + 1]
#             z21 = z[i3 + 1, i2, i1 + 1] + z[i3, i2, i1 + 1]
#             z31 = z[i3 + 1, i2 + 1, i1 + 1] + z[i3 + 1, i2, i1 + 1] + z11

#             u[2 * i3, 2 * i2, 2 * i1] = u[2 * i3, 2 * i2, 2 * i1] + z[i3, i2, i1]
#             u[2 * i3, 2 * i2, 2 * i1 + 1] = u[2 * i3, 2 * i2, 2 * i1 + 1] + 0.5 * (z[i3, i2, i1 + 1] + z[i3, i2, i1])
#             u[2 * i3, 2 * i2 + 1, 2 * i1] = u[2 * i3, 2 * i2 + 1, 2 * i1] + 0.5 * z10
#             u[2 * i3, 2 * i2 + 1, 2 * i1 + 1] = u[2 * i3, 2 * i2 + 1, 2 * i1 + 1] + 0.25 * (z10 + z11)
#             u[2 * i3 + 1, 2 * i2, 2 * i1] = u[2 * i3 + 1, 2 * i2, 2 * i1] + 0.5 * z20
#             u[2 * i3 + 1, 2 * i2, 2 * i1 + 1] = u[2 * i3 + 1, 2 * i2, 2 * i1 + 1] + 0.25 * (z20 + z21)
#             u[2 * i3 + 1, 2 * i2 + 1, 2 * i1] = u[2 * i3 + 1, 2 * i2 + 1, 2 * i1] + 0.25 * z30
#             u[2 * i3 + 1, 2 * i2 + 1, 2 * i1 + 1] = u[2 * i3 + 1, 2 * i2 + 1, 2 * i1 + 1] + 0.125 * (z30 + z31)
            
#             z10 = z11
#             z20 = z21
#             z30 = z31

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


# --------------------------------------------------------------------
# psinv applies an approximate inverse as smoother: u = u + Cr
#
# this  implementation costs  15A + 4M per result, where
# A and M denote the costs of Addition and Multiplication.
# presuming coefficient c(3) is zero (the NPB assumes this,
# but it is thus not a general case), 2A + 1M may be eliminated,
# resulting in 13A + 3M.
# note that this vectorizes, and is also fine for cache
# based machines.
# --------------------------------------------------------------------
#static void psinv(void* pointer_r, void* pointer_u, int n1, int n2, int n3, double c[4], int k)
@njit
def psinv(pointer_r, pointer_u, n1, n2, n3, c, k):
    #double (*r)[n2][n1] = (double (*)[n2][n1])pointer_r;
    #double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;
    r = pointer_r
    u = pointer_u

    r1 = numpy.empty(shape=M, dtype=numpy.float64)
    r2 = numpy.empty(shape=M, dtype=numpy.float64)

    #if timeron: Not supported by @njit
    #	c_timers.timer_start(T_PSINV)
    for i3 in range(1, n3 - 1):
        for i2 in range(1, n2 - 1):
            for i1 in range(n1):
                #r1[i1] = r[i3][i2-1][i1] + r[i3][i2+1][i1] + r[i3-1][i2][i1] + r[i3+1][i2][i1]
                r1[i1] = r[(i3 * n2 + (i2 - 1)) * n1 + i1] + r[(i3 * n2 + (i2 + 1)) * n1 + i1] + r[(
                    (i3 - 1) * n2 + i2) * n1 + i1] + r[((i3 + 1) * n2 + i2) * n1 + i1]
                #r2[i1] = r[i3-1][i2-1][i1] + r[i3-1][i2+1][i1] + r[i3+1][i2-1][i1] + r[i3+1][i2+1][i1]
                r2[i1] = r[((i3 - 1) * n2 + (i2 - 1)) * n1 + i1] + r[((i3 - 1) * n2 + (i2 + 1)) * n1 + i1] + r[(
                    (i3 + 1) * n2 + (i2 - 1)) * n1 + i1] + r[((i3 + 1) * n2 + (i2 + 1)) * n1 + i1]

            for i1 in range(1, n1 - 1):
                u[(i3 * n2 + i2) * n1 + i1] = (
                    u[(i3 * n2 + i2) * n1 + i1] + c[0] * r[(i3 * n2 + i2) * n1 + i1] + c[1] * (
                        r[(i3 * n2 + i2) * n1 + (i1 - 1)] + r[(i3 * n2 + i2) * n1 +
                                                              (i1 + 1)]  #( r[i3][i2][i1-1] + r[i3][i2][i1+1]
                        + r1[i1]) + c[2] * (r2[i1] + r1[i1 - 1] + r1[i1 + 1]))
                # --------------------------------------------------------------------
                # assume c(3) = 0    (enable line below if c(3) not= 0)
                # --------------------------------------------------------------------
                # > + c(3) * ( r2(i1-1) + r2(i1+1) )
                # --------------------------------------------------------------------
        #END
    #END
    #if timeron: Not supported by @njit
    #	c_timers.timer_stop(T_PSINV)

    # --------------------------------------------------------------------
    # exchange boundary points
    # --------------------------------------------------------------------
    comm3(u, n1, n2, n3, k)

    if debug_vec[0] >= 1:
        rep_nrm(u, n1, n2, n3, "   psinv", k)

    if debug_vec[3] >= k:
        showall(u, n1, n2, n3)


#END psinv()


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


# --------------------------------------------------------------------
# rprj3 projects onto the next coarser grid,
# using a trilinear finite element projection: s = r' = P r
#
# this  implementation costs 20A + 4M per result, where
# A and M denote the costs of addition and multiplication.
# note that this vectorizes, and is also fine for cache
# based machines.
# --------------------------------------------------------------------
#static void rprj3(void* pointer_r, int m1k, int m2k, int m3k, void* pointer_s, int m1j, int m2j, int m3j, int k)
@njit
def rprj3(pointer_r, m1k, m2k, m3k, pointer_s, m1j, m2j, m3j, k):
    #double (*r)[m2k][m1k] = (double (*)[m2k][m1k])pointer_r;
    #double (*s)[m2j][m1j] = (double (*)[m2j][m1j])pointer_s;
    r = pointer_r  #(i3*m2k+i2)*m1k + i1
    s = pointer_s  #(i3*m2j+i2)*m1j + i1

    x1 = numpy.empty(shape=M, dtype=numpy.float64)
    y1 = numpy.empty(shape=M, dtype=numpy.float64)

    #if timeron: Not supported by @njit
    #	c_timers.timer_start(T_RPRJ3)
    if m1k == 3:
        d1 = 2
    else:
        d1 = 1

    if m2k == 3:
        d2 = 2
    else:
        d2 = 1

    if m3k == 3:
        d3 = 2
    else:
        d3 = 1

    for j3 in range(1, m3j - 1):
        i3 = 2 * j3 - d3
        for j2 in range(1, m2j - 1):
            i2 = 2 * j2 - d2
            for j1 in range(1, m1j):
                i1 = 2 * j1 - d1
                #x1[i1] = r[i3+1][i2][i1] + r[i3+1][i2+2][i1] + r[i3][i2+1][i1] + r[i3+2][i2+1][i1]
                x1[i1] = r[((i3 + 1) * m2k + i2) * m1k + i1] + r[(
                    (i3 + 1) * m2k + (i2 + 2)) * m1k + i1] + r[(i3 * m2k +
                                                                (i2 + 1)) * m1k + i1] + r[((i3 + 2) * m2k +
                                                                                           (i2 + 1)) * m1k + i1]
                #y1[i1] = r[i3][i2][i1] + r[i3+2][i2][i1] + r[i3][i2+2][i1] + r[i3+2][i2+2][i1]
                y1[i1] = r[(i3 * m2k + i2) * m1k + i1] + r[(
                    (i3 + 2) * m2k + i2) * m1k + i1] + r[(i3 * m2k + (i2 + 2)) * m1k + i1] + r[((i3 + 2) * m2k +
                                                                                                (i2 + 2)) * m1k + i1]

            for j1 in range(1, m1j - 1):
                i1 = 2 * j1 - d1
                #y2 = r[i3][i2][i1+1] + r[i3+2][i2][i1+1] + r[i3][i2+2][i1+1] + r[i3+2][i2+2][i1+1]
                y2 = r[(i3 * m2k + i2) * m1k + (i1 + 1)] + r[((i3 + 2) * m2k + i2) * m1k +
                                                             (i1 + 1)] + r[(i3 * m2k + (i2 + 2)) * m1k +
                                                                           (i1 + 1)] + r[((i3 + 2) * m2k +
                                                                                          (i2 + 2)) * m1k + (i1 + 1)]
                #x2 = r[i3+1][i2][i1+1] + r[i3+1][i2+2][i1+1] + r[i3][i2+1][i1+1] + r[i3+2][i2+1][i1+1]
                x2 = r[((i3 + 1) * m2k + i2) * m1k +
                       (i1 + 1)] + r[((i3 + 1) * m2k + (i2 + 2)) * m1k +
                                     (i1 + 1)] + r[(i3 * m2k + (i2 + 1)) * m1k +
                                                   (i1 + 1)] + r[((i3 + 2) * m2k + (i2 + 1)) * m1k + (i1 + 1)]
                s[(j3 * m2j + j2) * m1j + j1] = (
                    0.5 * r[((i3 + 1) * m2k + (i2 + 1)) * m1k + (i1 + 1)]  #r[i3+1][i2+1][i1+1]
                    + 0.25 *
                    (r[((i3 + 1) * m2k +
                        (i2 + 1)) * m1k + i1] + r[((i3 + 1) * m2k + (i2 + 1)) * m1k +
                                                  (i1 + 2)] + x2)  #( r[i3+1][i2+1][i1] + r[i3+1][i2+1][i1+2] + x2) 
                    + 0.125 * (x1[i1] + x1[i1 + 2] + y2) + 0.0625 * (y1[i1] + y1[i1 + 2]))
        #END for j2 in range(1, m2j-1)
    #END for j3 in range(1, m3j-1)
    #if timeron: Not supported by @njit
    #	c_timers.timer_stop(T_RPRJ3)

    j = k - 1
    comm3(s, m1j, m2j, m3j, j)

    if debug_vec[0] >= 1:
        rep_nrm(s, m1j, m2j, m3j, "   rprj3", k - 1)

    if debug_vec[4] >= k:
        showall(s, m1j, m2j, m3j)


#END rprj3()


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


# --------------------------------------------------------------------
# multigrid v-cycle routine
# --------------------------------------------------------------------
#static void mg3P(double u[], double v[], double r[], double a[4], double c[4], int n1, int n2, int n3, int k)
@njit
def mg3P(u, v, r, a, c, n1, n2, n3, k):
    # --------------------------------------------------------------------
    # down cycle.
    # restrict the residual from the find grid to the coarse
    # -------------------------------------------------------------------
    for k in range(lt, lb + 1 - 1, -1):
        j = k - 1
        r_sub_k = r[ir[k]:]
        r_sub_j = r[ir[j]:]
        rprj3(r_sub_k, m1[k], m2[k], m3[k], r_sub_j, m1[j], m2[j], m3[j], k)

    k = lb
    # --------------------------------------------------------------------
    # compute an approximate solution on the coarsest grid
    # --------------------------------------------------------------------
    u_sub = u[ir[k]:]
    zero3(u_sub, m1[k], m2[k], m3[k])
    r_sub_p = r[ir[k]:]
    u_sub_p = u[ir[k]:]
    psinv(r_sub_p, u_sub_p, m1[k], m2[k], m3[k], c, k)

    for k in range(lb + 1, lt - 1 + 1):
        j = k - 1
        # --------------------------------------------------------------------
        # prolongate from level k-1  to k
        # -------------------------------------------------------------------
        u_sub1 = u[ir[k]:]
        zero3(u_sub1, m1[k], m2[k], m3[k])
        u_sub2_j = u[ir[j]:]
        u_sub2_k = u[ir[k]:]
        interp(u_sub2_j, m1[j], m2[j], m3[j], u_sub2_k, m1[k], m2[k], m3[k], k)
        # --------------------------------------------------------------------
        # compute residual for level k
        # --------------------------------------------------------------------
        u_sub3 = u[ir[k]:]
        r_sub3 = r[ir[k]:]
        resid(u_sub3, r_sub3, r_sub3, m1[k], m2[k], m3[k], a, k)
        # --------------------------------------------------------------------
        # apply smoother
        # --------------------------------------------------------------------
        r_sub4 = r[ir[k]:]
        u_sub4 = u[ir[k]:]
        psinv(r_sub4, u_sub4, m1[k], m2[k], m3[k], c, k)

    j = lt - 1
    k = lt
    u_sub_5 = u[ir[j]:]
    interp(u_sub_5, m1[j], m2[j], m3[j], u, n1, n2, n3, k)
    resid(u, v, r, n1, n2, n3, a, k)
    psinv(r, u, n1, n2, n3, c, k)


#END mg3P()


def mg3P_dace(u, v, r, a, c, n1, n2, n3, resid, rprj3, psinv, interp, combo=None):
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


#static void showall(void* pointer_z, int n1, int n2, int n3){
@njit
def showall(pointer_z, n1, n2, n3):
    #double (*z)[n2][n1] = (double (*)[n2][n1])pointer_z;
    z = pointer_z

    m1 = min(n1, 18)
    m2 = min(n2, 14)
    m3 = min(n3, 18)

    print()
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                print(z[(i3 * n2 + i2) * n1 + i1])  #Added break line to the original code
            print()
        print(" - - - - - - - ")
    print()


#END showall


# ---------------------------------------------------------------------
# report on norm
# ---------------------------------------------------------------------
#static void rep_nrm(void* pointer_u, int n1, int n2, int n3, char* title, int kk)
@njit
def rep_nrm(pointer_u, n1, n2, n3, title, kk):
    rnm2, rnmu = norm2u3(pointer_u, n1, n2, n3, nx[kk], ny[kk], nz[kk])
    print(" Level ", kk, " in ", title, ": norms =", rnm2, rnmu)


#END rep_nrm()


# --------------------------------------------------------------------
# resid computes the residual: r = v - Au
#
# this  implementation costs  15A + 4M per result, where
# A and M denote the costs of addition (or subtraction) and
# multiplication, respectively.
# presuming coefficient a(1) is zero (the NPB assumes this,
# but it is thus not a general case), 3A + 1M may be eliminated,
# resulting in 12A + 3M.
# note that this vectorizes, and is also fine for cache
# based machines.
# --------------------------------------------------------------------
#static void resid(void* pointer_u, void* pointer_v, void* pointer_r, int n1, int n2, int n3, double a[4], int k){
@njit
def resid(pointer_u, pointer_v, pointer_r, n1, n2, n3, a, k):
    #double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;
    #double (*v)[n2][n1] = (double (*)[n2][n1])pointer_v;
    #double (*r)[n2][n1] = (double (*)[n2][n1])pointer_r;
    u = pointer_u
    v = pointer_v
    r = pointer_r

    u1 = numpy.empty(shape=M, dtype=numpy.float64)
    u2 = numpy.empty(shape=M, dtype=numpy.float64)

    #if timeron: Not supported by @njit
    #	c_timers.timer_start(T_RESID)
    for i3 in range(1, n3 - 1):
        for i2 in range(1, n2 - 1):
            for i1 in range(n1):
                #u1[i1] = u[i3][i2-1][i1] + u[i3][i2+1][i1] + u[i3-1][i2][i1] + u[i3+1][i2][i1]
                u1[i1] = u[(i3 * n2 + (i2 - 1)) * n1 + i1] + u[(i3 * n2 + (i2 + 1)) * n1 + i1] + u[(
                    (i3 - 1) * n2 + i2) * n1 + i1] + u[((i3 + 1) * n2 + i2) * n1 + i1]
                #u2[i1] = u[i3-1][i2-1][i1] + u[i3-1][i2+1][i1] + u[i3+1][i2-1][i1] + u[i3+1][i2+1][i1]
                u2[i1] = u[((i3 - 1) * n2 + (i2 - 1)) * n1 + i1] + u[((i3 - 1) * n2 + (i2 + 1)) * n1 + i1] + u[(
                    (i3 + 1) * n2 + (i2 - 1)) * n1 + i1] + u[((i3 + 1) * n2 + (i2 + 1)) * n1 + i1]

            for i1 in range(1, n1 - 1):
                r[(i3 * n2 + i2) * n1 + i1] = (
                    v[(i3 * n2 + i2) * n1 + i1] - a[0] * u[(i3 * n2 + i2) * n1 + i1]
                    # ---------------------------------------------------------------------
                    # assume a(1) = 0 (enable 2 lines below if a(1) not= 0)
                    # ---------------------------------------------------------------------
                    # > - a(1) * ( u(i1-1,i2,i3) + u(i1+1,i2,i3)
                    # > + u1(i1) )
                    # ---------------------------------------------------------------------
                    - a[2] * (u2[i1] + u1[i1 - 1] + u1[i1 + 1]) - a[3] * (u2[i1 - 1] + u2[i1 + 1]))
        #END for i2 in range(1, n2-1):
    #END for i3 in range (1, n3-1):
    #if timeron: Not supported by @njit
    #	c_timers.timer_stop(T_RESID)

    # --------------------------------------------------------------------
    # exchange boundary data
    # --------------------------------------------------------------------
    comm3(r, n1, n2, n3, k)

    if debug_vec[0] >= 1:
        rep_nrm(r, n1, n2, n3, "   resid", k)

    if debug_vec[2] >= k:
        showall(r, n1, n2, n3)


#END resid()


@dace.program
def resid_dace(u: dace.float64[n0i, n0j, n0k], v: dace.float64[n0i, n0j, n0k], r: dace.float64[n0i, n0j, n0k],
               a: dace.float64[4]):

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


# ---------------------------------------------------------------------
# norm2u3 evaluates approximations to the l2 norm and the
# uniform (or l-infinity or chebyshev) norm, under the
# assumption that the boundaries are periodic or zero. add the
# boundaries in with half weight (quarter weight on the edges
# and eighth weight at the corners) for inhomogeneous boundaries.
# ---------------------------------------------------------------------
#static void norm2u3(void* pointer_r, int n1, int n2, int n3, double* rnm2, double* rnmu, int nx, int ny, int nz)
@njit
def norm2u3(pointer_r, n1, n2, n3, nx, ny, nz):
    #double (*r)[n2][n1] = (double (*)[n2][n1])pointer_r;
    r = pointer_r

    #if timeron: Not supported by @njit
    #	c_timers.timer_start(T_NORM2)
    dn = 1.0 * nx * ny * nz

    s = 0.0
    rnmu = 0.0
    for i3 in range(1, n3 - 1):
        for i2 in range(1, n2 - 1):
            for i1 in range(1, n1 - 1):
                s = s + r[(i3 * n2 + i2) * n1 + i1] * r[
                    (i3 * n2 + i2) * n1 + i1]  #s = s + r[i3][i2][i1] * r[i3][i2][i1]
                a = abs(r[(i3 * n2 + i2) * n1 + i1])  #a = abs(r[i3][i2][i1])
                if a > rnmu:
                    rnmu = a

    rnm2 = math.sqrt(s / dn)
    #if timeron: Not supported by @njit
    #	c_timers.timer_stop(T_NORM2)

    return rnm2, rnmu


#END norm2u3()


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


@dace.program
def combo_dace(z: dace.float64[n0i, n0j, n0k], u: dace.float64[n1i, n1j, n1k],
               v: dace.float64[n1i, n1j, n1k], r: dace.float64[n1i, n1j, n1k],
               a: dace.float64[4], c: dace.float64[4]):
    interp_dace(z, u)
    resid_dace(u, v, r, a)
    psinv_dace(r, u, c)


# ---------------------------------------------------------------------
# comm3 organizes the communication on all borders
# ---------------------------------------------------------------------
#static void comm3(void* pointer_u, int n1, int n2, int n3, int kk)
@njit
def comm3(pointer_u, n1, n2, n3, kk):
    #double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;
    u = pointer_u

    #if timeron: Not supported by @njit
    #	c_timers.timer_start(T_COMM3)

    # axis = 1
    for i3 in range(1, n3 - 1):
        for i2 in range(1, n2 - 1):
            u[(i3 * n2 + i2) * n1 + 0] = u[(i3 * n2 + i2) * n1 + (n1 - 2)]  #u[i3][i2][0] = u[i3][i2][n1-2]
            u[(i3 * n2 + i2) * n1 + (n1 - 1)] = u[(i3 * n2 + i2) * n1 + 1]  #u[i3][i2][n1-1] = u[i3][i2][1]

    # axis = 2
    for i3 in range(1, n3 - 1):
        for i1 in range(n1):
            u[(i3 * n2 + 0) * n1 + i1] = u[(i3 * n2 + (n2 - 2)) * n1 + i1]  #u[i3][0][i1] = u[i3][n2-2][i1]
            u[(i3 * n2 + (n2 - 1)) * n1 + i1] = u[(i3 * n2 + 1) * n1 + i1]  #u[i3][n2-1][i1] = u[i3][1][i1]

    # axis = 3
    for i2 in range(n2):
        for i1 in range(n1):
            (0 * n2 + i2) * n1 + i1
            u[(0 * n2 + i2) * n1 + i1] = u[((n3 - 2) * n2 + i2) * n1 + i1]  #u[0][i2][i1] = u[n3-2][i2][i1]
            u[((n3 - 1) * n2 + i2) * n1 + i1] = u[(1 * n2 + i2) * n1 + i1]  #u[n3-1][i2][i1] = u[1][i2][i1]

    #if timeron: Not supported by @njit
    #	c_timers.timer_stop(T_COMM3)


#END comm3()


# ---------------------------------------------------------------------
# bubble does a bubble sort in direction dir
# ---------------------------------------------------------------------
#static void bubble(double ten[][MM], int j1[][MM], int j2[][MM], int j3[][MM], int m, int ind)
@njit
def bubble(ten, j1, j2, j3, m, ind):
    if ind == 1:
        for i in range(m - 1):
            if ten[ind][i] > ten[ind][i + 1]:
                temp = ten[ind][i + 1]
                ten[ind][i + 1] = ten[ind][i]
                ten[ind][i] = temp

                j_temp = j1[ind][i + 1]
                j1[ind][i + 1] = j1[ind][i]
                j1[ind][i] = j_temp

                j_temp = j2[ind][i + 1]
                j2[ind][i + 1] = j2[ind][i]
                j2[ind][i] = j_temp

                j_temp = j3[ind][i + 1]
                j3[ind][i + 1] = j3[ind][i]
                j3[ind][i] = j_temp
            else:
                return
        #END for i in range(m-1):
    else:
        for i in range(m - 1):
            if ten[ind][i] < ten[ind][i + 1]:
                temp = ten[ind][i + 1]
                ten[ind][i + 1] = ten[ind][i]
                ten[ind][i] = temp

                j_temp = j1[ind][i + 1]
                j1[ind][i + 1] = j1[ind][i]
                j1[ind][i] = j_temp

                j_temp = j2[ind][i + 1]
                j2[ind][i + 1] = j2[ind][i]
                j2[ind][i] = j_temp

                j_temp = j3[ind][i + 1]
                j3[ind][i + 1] = j3[ind][i]
                j3[ind][i] = j_temp
            else:
                return
        #END for i in range(m-1):
    #END if ind == 1:


#END bubble()


# ---------------------------------------------------------------------
# power raises an integer, disguised as a double
# precision real, to an integer power
# ---------------------------------------------------------------------
#static double power(double a, int n)
@njit
def power(a, n):
    power = 1.0
    nj = n
    aj = a

    while nj != 0:
        if (nj % 2) == 1:
            rdummy, power = randlc(power, aj)
        rdummy, aj = randlc(aj, aj)
        nj = int(nj / 2)

    return power


#END power()


#---------------------------------------------------------------------
# zran3 loads +1 at ten randomly chosen points,
# loads -1 at a different ten random points,
# and zero elsewhere.
# ---------------------------------------------------------------------
#static void zran3(void* pointer_z, int n1, int n2, int n3, int nx, int ny, int k)
@njit
def zran3(pointer_z, n1, n2, n3, nx, ny, k):
    #double (*z)[n2][n1] = (double (*)[n2][n1])pointer_z;
    z = pointer_z

    ten = numpy.empty(shape=(2, MM), dtype=numpy.float64)
    j1 = numpy.empty(shape=(2, MM), dtype=numpy.int32)
    j2 = numpy.empty(shape=(2, MM), dtype=numpy.int32)
    j3 = numpy.empty(shape=(2, MM), dtype=numpy.int32)
    jg = numpy.empty(shape=(2, MM, 4), dtype=numpy.int32)

    a1 = power(A, nx)
    a2 = power(A, nx * ny)

    zero3(z, n1, n2, n3)

    i = int(is1 - 2 + nx * (is2 - 2 + ny * (is3 - 2)))

    ai = power(A, i)
    d1 = ie1 - is1 + 1
    e1 = ie1 - is1 + 2
    e2 = ie2 - is2 + 2
    e3 = ie3 - is3 + 2
    x0 = X
    aux, x0 = randlc(x0, ai)
    for i3 in range(1, e3):
        x1 = x0
        for i2 in range(1, e2):
            xx = x1

            idx = (i3 * n2 + i2) * n1 + 1
            z_sub = z[idx:]
            xx = vranlc(d1, xx, A, z_sub)  #vranlc(d1, &xx, A, &(z[i3][i2][1]));

            aux, x1 = randlc(x1, a1)
        aux, x0 = randlc(x0, a2)

    # ---------------------------------------------------------------------
    # each processor looks for twenty candidates
    # ---------------------------------------------------------------------
    # for i in range(MM):
    #     ten[1][i] = 0.0
    #     j1[1][i] = 0
    #     j2[1][i] = 0
    #     j3[1][i] = 0
    #     ten[0][i] = 1.0
    #     j1[0][i] = 0
    #     j2[0][i] = 0
    #     j3[0][i] = 0
    ten[0][:MM] = 1.0
    ten[1][:MM] = 0.0
    j1[:][:MM] = 0
    j2[:][:MM] = 0
    j3[:][:MM] = 0

    for i3 in range(1, n3 - 1):
        for i2 in range(1, n2 - 1):
            for i1 in range(1, n1 - 1):
                if z[(i3 * n2 + i2) * n1 + i1] > ten[1][0]:  #if(z[i3][i2][i1] > ten[1][0]){
                    ten[1][0] = z[(i3 * n2 + i2) * n1 + i1]
                    j1[1][0] = i1
                    j2[1][0] = i2
                    j3[1][0] = i3
                    bubble(ten, j1, j2, j3, MM, 1)

                if z[(i3 * n2 + i2) * n1 + i1] < ten[0][0]:  #if(z[i3][i2][i1] < ten[0][0]){
                    ten[0][0] = z[(i3 * n2 + i2) * n1 + i1]
                    j1[0][0] = i1
                    j2[0][0] = i2
                    j3[0][0] = i3
                    bubble(ten, j1, j2, j3, MM, 0)
    #END for i3 in range(1, n3-1):

    # ---------------------------------------------------------------------
    # now which of these are globally best?
    # ---------------------------------------------------------------------
    i1 = MM - 1
    i0 = MM - 1
    for i in range(MM - 1, 0 - 1, -1):
        best = 0.0
        if best < ten[1][i1]:
            jg[1][i][0] = 0
            jg[1][i][1] = is1 - 2 + j1[1][i1]
            jg[1][i][2] = is2 - 2 + j2[1][i1]
            jg[1][i][3] = is3 - 2 + j3[1][i1]
            i1 = i1 - 1
        else:
            jg[1][i][0] = 0
            jg[1][i][1] = 0
            jg[1][i][2] = 0
            jg[1][i][3] = 0

        best = 1.0
        if best > ten[0][i0]:
            jg[0][i][0] = 0
            jg[0][i][1] = is1 - 2 + j1[0][i0]
            jg[0][i][2] = is2 - 2 + j2[0][i0]
            jg[0][i][3] = is3 - 2 + j3[0][i0]
            i0 = i0 - 1
        else:
            jg[0][i][0] = 0
            jg[0][i][1] = 0
            jg[0][i][2] = 0
            jg[0][i][3] = 0
    #END for i in range(MM-1, 0-1, -1):
    m1 = 0
    m0 = 0

    # for i3 in range(n3):
    #     for i2 in range(n2):
    #         for i1 in range(n1):
    #             z[(i3 * n2 + i2) * n1 + i1] = 0.0  #z[i3][i2][i1] = 0.0
    z[:n1 * n2 * n3] = 0.0

    for i in range(MM - 1, m0 - 1, -1):
        #z[jg[0][i][3]][jg[0][i][2]][jg[0][i][1]] = -1.0
        i3, i2, i1 = jg[0][i][3], jg[0][i][2], jg[0][i][1]
        z[(i3 * n2 + i2) * n1 + i1] = -1.0

    for i in range(MM - 1, m1 - 1, -1):
        #z[jg[1][i][3]][jg[1][i][2]][jg[1][i][1]] = +1.0
        i3, i2, i1 = jg[1][i][3], jg[1][i][2], jg[1][i][1]
        z[(i3 * n2 + i2) * n1 + i1] = +1.0

    comm3(z, n1, n2, n3, k)


#END zran3()


@njit
def zero3(pointer_z, n1, n2, n3):
    #double (*z)[n2][n1] = (double (*)[n2][n1])pointer_z;
    z = pointer_z

    # for i3 in range(n3):
    #     for i2 in range(n2):
    #         for i1 in range(n1):
    #             z[(i3 * n2 + i2) * n1 + i1] = 0.0
    z[:n1*n2*n3] = 0.0


#END zero3()


@njit
def setup(k, nx, ny, nz, m1, m2, m3, ir):
    mi = numpy.empty(shape=(MAXLEVEL + 1, 3), dtype=numpy.int32)
    ng = numpy.empty(shape=(MAXLEVEL + 1, 3), dtype=numpy.int32)

    ng[lt][0] = nx[lt]
    ng[lt][1] = ny[lt]
    ng[lt][2] = nz[lt]
    for ax in range(3):
        for k in range(lt - 1, 1 - 1, -1):
            ng[k][ax] = ng[k + 1][ax] / 2

    for k in range(lt, 1 - 1, -1):
        nx[k] = ng[k][0]
        ny[k] = ng[k][1]
        nz[k] = ng[k][2]

    for k in range(lt, 1 - 1, -1):
        for ax in range(3):
            mi[k][ax] = 2 + ng[k][ax]

        m1[k] = mi[k][0]
        m2[k] = mi[k][1]
        m3[k] = mi[k][2]

    k = lt
    is1 = 2 + ng[k][0] - ng[lt][0]
    ie1 = 1 + ng[k][0]
    n1 = 3 + ie1 - is1
    is2 = 2 + ng[k][1] - ng[lt][1]
    ie2 = 1 + ng[k][1]
    n2 = 3 + ie2 - is2
    is3 = 2 + ng[k][2] - ng[lt][2]
    ie3 = 1 + ng[k][2]
    n3 = 3 + ie3 - is3

    ir[lt] = 0

    for j in range(lt - 1, 1 - 1, -1):
        ir[j] = ir[j + 1] + npbparams.ONE * m1[j + 1] * m2[j + 1] * m3[j + 1]

    if debug_vec[1] >= 1:
        print(" in setup")
        print("   k  lt  nx  ny  nz  n1  n2  n3 is1 is2 is3 ie1 ie2 ie3")
        print("  ", k, " ", lt, "", ng[k][0], "", ng[k][1], "", ng[k][2], "", n1, "", n2, "", n3, " ", is1, " ", is2,
              " ", is3, "", ie1, "", ie2, "", ie3)

    return n1, n2, n3, is1, is2, is3, ie1, ie2, ie3


#END setup()


def main(framework: str):

    import time

    runtime = -time.perf_counter()

    global nx, ny, nz, m1, m2, m3, ir
    global u, v, r
    global debug_vec
    global is1, is2, is3, ie1, ie2, ie3, lt, lb
    global timeron

    for i in range(T_LAST):
        c_timers.timer_clear(i)

    t_names = numpy.empty(T_LAST, dtype=object)

    c_timers.timer_start(T_INIT)

    # timeron = os.path.isfile("timer.flag")
    timeron = True
    if timeron:
        t_names[T_INIT] = "init"
        t_names[T_BENCH] = "benchmk"
        t_names[T_MG3P] = "mg3P"
        t_names[T_PSINV] = "psinv*"
        t_names[T_RESID] = "resid*"
        t_names[T_RPRJ3] = "rprj3*"
        t_names[T_INTERP] = "interp*"
        t_names[T_NORM2] = "norm2*"
        t_names[T_COMM3] = "comm3*"

    fp = os.path.isfile("mg.input")
    if fp:
        print(" Reading from input file mg.input")
        print(" ERROR - Not implemented")
        sys.exit()
    else:
        print(" No input file. Using NPB class defaults")
        lt = npbparams.LT_DEFAULT
        nit = npbparams.NIT_DEFAULT
        nx[lt] = npbparams.NX_DEFAULT
        ny[lt] = npbparams.NY_DEFAULT
        nz[lt] = npbparams.NZ_DEFAULT
        #for i in range (7+1): #Already innitialized
        #	debug_vec[i] = npbparams.DEBUG_DEFAULT
    
    def gpu_storage(sdfg: dace.SDFG):
        for _, desc in sdfg.arrays.items():
            if not desc.transient and isinstance(desc, dace.data.Array):
                desc.storage = dace.StorageType.GPU_Global
    
    def compile_program_cpu(prog: dace.program, use_stencil_tiling: bool = True):
        sdfg = prog.to_sdfg(simplify=False)
        sdfg.simplify()
        auto_optimize(sdfg, dace.DeviceType.CPU, use_stencil_tiling=use_stencil_tiling)
        # sdfg.apply_transformations_repeated([TaskletFusion])
        # sdfg.apply_transformations([Vectorization], options={'vector_len': 2})
        return sdfg.compile()
    
    def compile_program_gpu(prog: dace.program, use_stencil_tiling: bool = True, use_wcr_tiling: bool = True):  
        sdfg = prog.to_sdfg(simplify=False)
        sdfg.simplify()
        auto_optimize(sdfg, dace.DeviceType.GPU, use_stencil_tiling=use_stencil_tiling, use_wcr_tiling=use_wcr_tiling, use_gpu_storage=True)
        return sdfg.compile()
    
    combo_func = None

    runtime += time.perf_counter()
    print(" Initialization time: %f seconds" % runtime)

    runtime = -time.perf_counter()

    # if framework == 'dace_cpu':
    #     print("Compiling DaCe versions")
    #     resid_func = compile_program_cpu(resid_dace)
    #     rprj3_func = compile_program_cpu(rprj3_dace)
    #     psinv_func = compile_program_cpu(psinv_dace)
    #     interp_func = compile_program_cpu(interp_dace, use_stencil_tiling=False)
    #     combo_func = compile_program_cpu(combo_dace, use_stencil_tiling=False)
    #     # sdfg = interp_dace.to_sdfg(simplify=True)  # Issue with persistent z1, z2, z3
    #     # interp_func = sdfg.compile()
    #     # sdfg = norm2u3_dace.to_sdfg(simplify=True)
    #     # auto_optimize(sdfg, dace.DeviceType.CPU)
    #     # sdfg.simplify()
    #     # auto_optimize(sdfg, dace.DeviceType.CPU)
    #     # norm2u3_func = compile_program_cpu(norm2u3_dace)
    #     sdfg = norm2u3_dace.to_sdfg(simplify=True)
    #     sdfg.apply_transformations_repeated([MapReduceFusion])
    #     tile_wcrs(sdfg, validate_all=True)
    #     greedy_fuse(sdfg, validate_all=True)
    #     norm2u3_func = sdfg.compile()  # Custom workflow to work around inability to tile two WCRs in a single Map
    # elif framework == 'dace_gpu':
    #     print("Compiling DaCe versions")
    #     resid_func = compile_program_gpu(resid_dace)
    #     rprj3_func = compile_program_gpu(rprj3_dace)
    #     psinv_func = compile_program_gpu(psinv_dace)
    #     combo_func = compile_program_gpu(combo_dace, use_stencil_tiling=False)
    #     interp_func = compile_program_gpu(interp_dace, use_stencil_tiling=False)
    #     # sdfg = interp_dace.to_sdfg(simplify=True)
    #     # gpu_storage(sdfg)
    #     # sdfg.apply_gpu_transformations()
    #     # interp_func = sdfg.compile() # Issue with persistent z1, z2, z3
    #     norm2u3_func = compile_program_gpu(norm2u3_dace, use_wcr_tiling=False)

    resid_func, rprj3_func, psinv_func, interp_func, combo_func, norm2u3_func = get_mg_dace(framework)
    
    runtime += time.perf_counter()
    print(" Compilation time: %f seconds" % runtime)

    runtime = -time.perf_counter()

    # ---------------------------------------------------------------------
    # use these for debug info:
    # ---------------------------------------------------------------------
    # debug_vec(0) = 1 !=> report all norms
    # debug_vec(1) = 1 !=> some setup information
    # debug_vec(1) = 2 !=> more setup information
    # debug_vec(2) = k => at level k or below, show result of resid
    # debug_vec(3) = k => at level k or below, show result of psinv
    # debug_vec(4) = k => at level k or below, show result of rprj
    # debug_vec(5) = k => at level k or below, show result of interp
    # debug_vec(6) = 1 => (unused)
    # debug_vec(7) = 1 => (unused)
    # ---------------------------------------------------------------------
    a = numpy.empty(4, dtype=numpy.float64)
    a[0] = -8.0 / 3.0
    a[1] = 0.0
    a[2] = 1.0 / 6.0
    a[3] = 1.0 / 12.0

    c = numpy.empty(4, dtype=numpy.float64)
    if npbparams.CLASS in ['A', 'S', 'W']:
        # coefficients for the s(a) smoother
        c[0] = -3.0 / 8.0
        c[1] = +1.0 / 32.0
        c[2] = -1.0 / 64.0
        c[3] = 0.0
    else:
        # coefficients for the s(b) smoother
        c[0] = -3.0 / 17.0
        c[1] = +1.0 / 33.0
        c[2] = -1.0 / 61.0
        c[3] = 0.0

    lb = 1
    k = lt

    n1, n2, n3, is1, is2, is3, ie1, ie2, ie3 = setup(k, nx, ny, nz, m1, m2, m3, ir)

    runtime += time.perf_counter()
    print(" Setup time: %f seconds" % runtime)

    runtime = -time.perf_counter()

    # zero3(u, n1, n2, n3)
    # zran3(v, n1, n2, n3, nx[lt], ny[lt], k)

    # rnm2, rnmu = norm2u3(v, n1, n2, n3, nx[lt], ny[lt], nz[lt])


    print("\n\n NAS Parallel Benchmarks 4.1 Serial Python version - MG Benchmark\n")
    print(" Size: %3dx%3dx%3d (class_npb %1c)" % (nx[lt], ny[lt], nz[lt], npbparams.CLASS))
    print(" Iterations: %3d" % (nit))

    # resid(u, v, r, n1, n2, n3, a, k)
    # rnm2, rnmu = norm2u3(r, n1, n2, n3, nx[lt], ny[lt], nz[lt])

    if framework == "python":

        # ---------------------------------------------------------------------
        # one iteration for startup
        # ---------------------------------------------------------------------
        mg3P(u, v, r, a, c, n1, n2, n3, k)
        resid(u, v, r, n1, n2, n3, a, k)

        n1, n2, n3, is1, is2, is3, ie1, ie2, ie3 = setup(k, nx, ny, nz, m1, m2, m3, ir)

        zero3(u, n1, n2, n3)
        zran3(v, n1, n2, n3, nx[lt], ny[lt], k)

        c_timers.timer_stop(T_INIT)
        tinit = c_timers.timer_read(T_INIT)
        print(" Initialization time: %15.3f seconds" % (tinit))

        for i in (T_BENCH, T_LAST):
            c_timers.timer_clear(i)
        c_timers.timer_start(T_BENCH)

        if timeron:
            c_timers.timer_start(T_RESID2)
        resid(u, v, r, n1, n2, n3, a, k)
        if timeron:
            c_timers.timer_stop(T_RESID2)
        rnm2, rnmu = norm2u3(r, n1, n2, n3, nx[lt], ny[lt], nz[lt])

        for it in range(1, nit + 1):
            if it == 1 or it == nit or (it % 5) == 0:
                print("  iter %3d" % (it))
            if timeron:
                c_timers.timer_start(T_MG3P)
            mg3P(u, v, r, a, c, n1, n2, n3, k)
            if timeron:
                c_timers.timer_stop(T_MG3P)
            if timeron:
                c_timers.timer_start(T_RESID2)
            resid(u, v, r, n1, n2, n3, a, k)
            if timeron:
                c_timers.timer_stop(T_RESID2)

        rnm2, rnmu = norm2u3(r, n1, n2, n3, nx[lt], ny[lt], nz[lt])

        c_timers.timer_stop(T_BENCH)
        t = c_timers.timer_read(T_BENCH)
    
    elif framework in ("dace_cpu", "dace_gpu"):

        if framework == "dace_gpu":
            du = cupy.asarray(u)
            dv = cupy.asarray(v)
            dr = cupy.asarray(r)
            a = cupy.asarray(a)
            c = cupy.asarray(c)
            cupy.cuda.runtime.deviceSynchronize()
        else:
            du = u
            dv = v
            dr = r

        my_u = du[:n1*n2*n3].reshape(n3, n2, n1)
        my_v = dv[:n1*n2*n3].reshape(n3, n2, n1)
        my_r = dr[:n1*n2*n3].reshape(n3, n2, n1)
        dn = nx[lt]*ny[lt]*nz[lt]
 
        # ---------------------------------------------------------------------
        # one iteration for startup
        # ---------------------------------------------------------------------
        mg3P_dace(du, dv, dr, a, c, n1, n2, n3, resid_func, rprj3_func, psinv_func, interp_func, combo_func)
        cupy.cuda.runtime.deviceSynchronize()
        resid_func(u=my_u, v=my_v, r=my_r, a=a, n0i=n3, n0j=n2, n0k=n1)
        cupy.cuda.runtime.deviceSynchronize()
        norm2u3_func(r=my_r, dn=dn, n0i=n3, n0j=n2, n0k=n1)
        cupy.cuda.runtime.deviceSynchronize()

        runtime += time.perf_counter()
        print(" Warm-up time: %f seconds" % runtime)

        runtime = -time.perf_counter()
    
        n1, n2, n3, is1, is2, is3, ie1, ie2, ie3 = setup(k, nx, ny, nz, m1, m2, m3, ir)

        runtime += time.perf_counter()
        print(" Setup time: %f seconds" % runtime)

        runtime = -time.perf_counter()

        zero3(u, n1, n2, n3)
        runtime += time.perf_counter()
        print(" Zero time: %f seconds" % runtime)
        runtime = -time.perf_counter()
        zran3(v, n1, n2, n3, nx[lt], ny[lt], k)
        runtime += time.perf_counter()
        print(" Zran time: %f seconds" % runtime)

        if framework == "dace_gpu":
            du[:] = cupy.asarray(u)
            dv[:] = cupy.asarray(v)
            dr[:] = cupy.asarray(r)
            cupy.cuda.runtime.deviceSynchronize()

        c_timers.timer_stop(T_INIT)
        tinit = c_timers.timer_read(T_INIT)
        print(" Initialization time: %15.3f seconds" % (tinit))

        for i in (T_BENCH, T_LAST):
            c_timers.timer_clear(i)
        c_timers.timer_start(T_BENCH)

        # if timeron:
        #     c_timers.timer_start(T_RESID2)
        # resid_func(u=my_u, v=my_v, r=my_r, a=a, n0i=n3, n0j=n2, n0k=n1)
        # # cupy.cuda.runtime.deviceSynchronize()

        # if timeron:
        #     c_timers.timer_stop(T_RESID2)
        # rnm2, rnmu = norm2u3_func(r=my_r, dn=dn, n0i=n3, n0j=n2, n0k=n1)
        # # cupy.cuda.runtime.deviceSynchronize()

        # for it in range(1, nit+1):
        #     if it == 1 or it == nit or (it%5) == 0:
        #         print("  iter %3d" % (it))
        #     if timeron:
        #         c_timers.timer_start(T_MG3P)
        #     mg3P_dace(du, dv, dr, a, c, n1, n2, n3, resid_func, rprj3_func, psinv_func, interp_func, combo_func)
        #     # cupy.cuda.runtime.deviceSynchronize()
        #     if timeron:
        #         c_timers.timer_stop(T_MG3P)
        #     if timeron:
        #         c_timers.timer_start(T_RESID2)
        #     resid_func(u=my_u, v=my_v, r=my_r, a=a, n0i=n3, n0j=n2, n0k=n1)
        #     # cupy.cuda.runtime.deviceSynchronize()

        #     if timeron:
        #         c_timers.timer_stop(T_RESID2)

        # rnm2, rnmu = norm2u3_func(r=my_r, dn=dn, n0i=n3, n0j=n2, n0k=n1)
        # # cupy.cuda.runtime.deviceSynchronize()

        mg_dace_full(lt, lb, m1, m2, m3, ir, dn, nit, du, dv, dr, a, c, n1, n2, n3, resid_func, norm2u3_func, rprj3_func, psinv_func, interp_func, combo_func=combo_func)

        c_timers.timer_stop(T_BENCH)
        t = c_timers.timer_read(T_BENCH)

        if framework == "dace_gpu":
            rnm2 = cupy.asnumpy(rnm2)

    verify_value = 0.0

    print(" Benchmark completed")

    epsilon = 1.0e-8

    if npbparams.CLASS == 'S':
        verify_value = 0.5307707005734e-04
    elif npbparams.CLASS == 'W':
        verify_value = 0.6467329375339e-05
    elif npbparams.CLASS == 'A':
        verify_value = 0.2433365309069e-05
    elif npbparams.CLASS == 'B':
        verify_value = 0.1800564401355e-05
    elif npbparams.CLASS == 'C':
        verify_value = 0.5706732285740e-06
    elif npbparams.CLASS == 'D':
        verify_value = 0.1583275060440e-09
    elif npbparams.CLASS == 'E':
        verify_value = 0.8157592357404e-10

    verified = False
    err = abs(rnm2 - verify_value) / verify_value
    if err <= epsilon:
        verified = True
        print(" VERIFICATION SUCCESSFUL")
        print(" L2 Norm is %20.13e" % (rnm2))
        print(" Error is   %20.13e" % (err))
    else:
        print(" VERIFICATION FAILED")
        print(" L2 Norm is             %20.13e" % (rnm2))
        print(" The correct L2 Norm is %20.13e" % (err))

    nn = 1.0 * nx[lt] * ny[lt] * nz[lt]

    mflops = 0.0
    if t != 0.0:
        mflops = 58.0 * nit * nn * 1.0e-6 / t

    c_print_results.c_print_results("MG", npbparams.CLASS, nx[lt], ny[lt], nz[lt], nit, t, mflops,
                                    "          floating point", verified)

    # ---------------------------------------------------------------------
    # more timers
    # ---------------------------------------------------------------------
    if timeron:
        tmax = c_timers.timer_read(T_BENCH)
        if tmax == 0.0:
            tmax = 1.0
        print("  SECTION   Time (secs)")
        for i in range(T_BENCH, T_LAST):
            t = c_timers.timer_read(i)
            if i == T_RESID2:
                t = c_timers.timer_read(T_RESID) - t
                print("    --> %8s:%9.3f  (%6.2f%%)" % ("mg-resid", t, t * 100.0 / tmax))
            else:
                print("  %-8s:%9.3f  (%6.2f%%)" % (t_names[i], t, t * 100.0 / tmax))
        print("  (* Time hasn't gauged: operation is not supported by @njit)")
    
    def _func():
        resid_func(u=my_u, v=my_v, r=my_r, a=a, n0i=n3, n0j=n2, n0k=n1)
        rnm2, rnmu = norm2u3_func(r=my_r, dn=dn, n0i=n3, n0j=n2, n0k=n1)
        for it in range(1, nit+1):
            mg3P_dace(du, dv, dr, a, c, n1, n2, n3, resid_func, rprj3_func, psinv_func, interp_func, combo_func)
            resid_func(u=my_u, v=my_v, r=my_r, a=a, n0i=n3, n0j=n2, n0k=n1)
        rnm2, rnmu = norm2u3_func(r=my_r, dn=dn, n0i=n3, n0j=n2, n0k=n1)
    
    from timeit import repeat
    num_warmup = 2
    for _ in range(num_warmup):
        _func()
    runtimes = repeat(lambda: _func(), number=1, repeat=10)
    mean = numpy.mean(runtimes)
    std = numpy.std(runtimes)
    repeatitions = 10
    while std > 0.01 * mean and len(runtimes) < 100:
        print(f"Standard deviation too high ({std * 100 / mean:.2f}% of the mean) after {repeatitions} repeatitions ...", flush=True)
        runtimes.extend(repeat(lambda: _func(), number=1, repeat=10))
        mean = numpy.mean(runtimes)
        std = numpy.std(runtimes)
        repeatitions += 10
    mflops = 58.0 * nit * nn * 1.0e-6 / mean
    print(f"DaCe {'CPU' if args.framework == 'dace_cpu' else 'GPU'} runtime: mean {mean} s ({mflops} mflop/s), std {std * 100 / mean:.2f}%", flush=True)


#END main()

#Starting of execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NPB-PYTHON-SER MG')
    parser.add_argument("-c", "--CLASS", default='A', help="WORKLOADs CLASSes")
    parser.add_argument("-f", "--framework", choices=['python', 'dace_cpu', 'dace_gpu'], default='dace_cpu', help="Framework to use")
    args = parser.parse_args()

    npbparams.set_mg_info(args.CLASS)
    set_global_variables()

    main(args.framework)
