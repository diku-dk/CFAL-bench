"""
    Testbench for rollback, tridag, and the entire main loop (updateParams + rollback)
"""

import dace

import numpy as np
import math
import time
from dace.transformation.auto.auto_optimize import auto_optimize
import copy

numX, numY, numT, N, Ntridiag, numZ = (dace.symbol(s, dtype=dace.int32)
                                       for s in ('numX', 'numY', 'numT', 'N', 'Ntridiag', 'numZ'))

Tvalue = dace.symbol("Tvalue", dtype=dace.float64)


@dace.program
def updateParams(g: int, alpha: float, beta: float, nu: float, X: dace.float64[numX], Y: dace.float64[numY],
                 Time: dace.float64[numT], MuX: dace.float64[numY, numX], VarX: dace.float64[numY, numX],
                 MuY: dace.float64[numX, numY], VarY: dace.float64[numX, numY]):

    # DaCE
    # Maybe can be improved for readability/functional-friendliness
    # - second loop nest can be avoided but currently we are missing fill support
    # - the first one maybe can be further simplified?

    # MuX.fill(0)
    # for j in range(numY):
    #     VarX[j, :] = np.exp(2 * (beta * np.log(X) + Y[j] - 0.5 * nu * nu * Time[g]))

    for j in range(numY):
        for i in range(numX):
            MuX[j, i] = 0.0
            # VarX[j, i] = np.exp(
            #     2 * (beta * np.log(X[i]) + Y[j] - 0.5 * nu * nu * Time[g])
            # )  # TODO: original line: Time[g] was causing problems with free symbols if auto-opt are used
            VarX[j, i] = np.exp(2 * (beta * np.log(X[i]) + Y[j] - 0.5 * nu * nu * Tvalue))

    # MuY.fill(0)
    # VarY.fill(nu**2)  # apparantely currently not supported in DaCe is not ok
    for i in range(numX):
        for j in range(numY):
            MuY[i, j] = 0.0
            VarY[i, j] = nu * nu


def tridag(Ntridiag, a: dace.float64[Ntridiag], b: dace.float64[Ntridiag], c: dace.float64[Ntridiag],
           y: dace.float64[Ntridiag]):
    """
    Computes the solution of the tridiagonal system in output array y
    - a contains the N-1 subdiagonal elements
    - b contains the N diagonal elements
    - c contains the N-1 superdiagonal elements
    - y right hand side and output
    """

    # DaCe TODOs:
    # - currently using a different symbol (Ntridiag vs N). Maybe we can use another symbol (numZ is the max between numX and numY)
    # - rewrite in a more array-programminsh way

    #forward swap
    for i in range(1, Ntridiag - 1):
        beta = a[i] / b[i - 1]

        b[i] = b[i] - beta * c[i - 1]
        y[i] = y[i] - beta * y[i - 1]

    #backward
    y[Ntridiag - 1] = y[Ntridiag - 1] / b[Ntridiag - 1]
    for i in range(Ntridiag - 2, -1, -1):
        y[i] = (y[i] - c[i] * y[i + 1]) / b[i]


@dace.program
def dace_tridag(a: dace.float64[Ntridiag], b: dace.float64[Ntridiag], c: dace.float64[Ntridiag],
                y: dace.float64[Ntridiag]):

    #forward swap
    for i in range(1, Ntridiag - 1):
        beta = a[i] / b[i - 1]

        b[i] = b[i] - beta * c[i - 1]
        y[i] = y[i] - beta * y[i - 1]

    #backward
    y[Ntridiag - 1] = y[Ntridiag - 1] / b[Ntridiag - 1]
    for i in range(Ntridiag - 2, -1, -1):
        y[i] = (y[i] - c[i] * y[i + 1]) / b[i]


def rollback(g: int, a: dace.float64[numZ], b: dace.float64[numZ], c: dace.float64[numZ], Time: dace.float64[numT],
             U: dace.float64[numY, numX], V: dace.float64[numX, numY], Dx: dace.float64[numX,
                                                                                        3], Dxx: dace.float64[numX, 3],
             MuX: dace.float64[numY, numX], VarX: dace.float64[numY, numX], Dy: dace.float64[numY,
                                                                                             3], Dyy: dace.float64[numY,
                                                                                                                   3],
             MuY: dace.float64[numX, numY], VarY: dace.float64[numX, numY], ResultE: dace.float64[numX, numY]):

    dtInv = 1.0 / (Time[g + 1] - Time[g])

    # tridag_sdfg = tridag.to_sdfg()

    # TODOs:
    # - With Tridag we are passing a view. This generates a TypeError: Passing a numpy view ...

    # explicit x
    for j in range(numY):
        for i in range(numX):
            U[j, i] = dtInv * ResultE[i, j]

            if i > 0:
                U[j, i] += 0.5 * ResultE[i - 1, j] * (MuX[j, i] * Dx[i, 0] + 0.5 * VarX[j, i] * Dxx[i, 0])

            U[j, i] += 0.5 * ResultE[i, j] * (MuX[j, i] * Dx[i, 1] + 0.5 * VarX[j, i] * Dxx[i, 1])

            if i < numX - 1:
                U[j, i] += 0.5 * ResultE[i + 1, j] * (MuX[j, i] * Dx[i, 2] + 0.5 * VarX[j, i] * Dxx[i, 2])

    # explicit y
    for i in range(numX):
        for j in range(numY):

            V[i, j] = 0.0
            if j > 0:
                V[i, j] += ResultE[i, j - 1] * (MuY[i, j] * Dy[j, 0] + 0.5 * VarY[i, j] * Dyy[j, 0])
            V[i, j] += ResultE[i, j] * (MuY[i, j] * Dy[j, 1] + 0.5 * VarY[i, j] * Dyy[j, 1])
            if j < numY - 1:
                V[i, j] += ResultE[i, j + 1] * (MuY[i, j] * Dy[j, 2] + 0.5 * VarY[i, j] * Dyy[j, 2])

            U[j, i] += V[i, j]

    # implicit x
    for j in range(numY):
        for i in range(numX):
            a[i] = -0.5 * (MuX[j, i] * Dx[i, 0] + 0.5 * VarX[j, i] * Dxx[i, 0])
            b[i] = dtInv - 0.5 * (MuX[j, i] * Dx[i, 1] + 0.5 * VarX[j, i] * Dxx[i, 1])
            c[i] = -0.5 * (MuX[j, i] * Dx[i, 2] + 0.5 * VarX[j, i] * Dxx[i, 2])

        uu = U[j, :]

        tridag(numX, a, b, c, uu)
        # tridag(a, b, c, uu, Ntridiag=numX)

    # implicit y
    for i in range(numX):
        for j in range(numY):
            a[j] = -0.5 * (MuY[i, j] * Dy[j, 0] + 0.5 * VarY[i, j] * Dyy[j, 0])
            b[j] = dtInv - 0.5 * (MuY[i, j] * Dy[j, 1] + 0.5 * VarY[i, j] * Dyy[j, 1])
            c[j] = -0.5 * (MuY[i, j] * Dy[j, 2] + 0.5 * VarY[i, j] * Dyy[j, 2])

        yy = ResultE[i, :]

        for j in range(numY):
            yy[j] = dtInv * U[j, i] - 0.5 * V[i, j]

        tridag(
            numY,
            a,
            b,
            c,
            yy,
        )


@dace.program
def rollback_dace(g: int, Time: dace.float64[numT], U: dace.float64[numY, numX], V: dace.float64[numX, numY],
                  Dx: dace.float64[numX, 3], Dxx: dace.float64[numX, 3], MuX: dace.float64[numY, numX],
                  VarX: dace.float64[numY, numX], Dy: dace.float64[numY, 3], Dyy: dace.float64[numY, 3],
                  MuY: dace.float64[numX, numY], VarY: dace.float64[numX, numY], ResultE: dace.float64[numX, numY]):

    dtInv = 1.0 / (Time[g + 1] - Time[g])

    # explicit x
    for j in range(numY):
        for i in range(numX):
            U[j, i] = dtInv * ResultE[i, j]

            if i > 0:
                U[j, i] += 0.5 * ResultE[i - 1, j] * (MuX[j, i] * Dx[i, 0] + 0.5 * VarX[j, i] * Dxx[i, 0])

            U[j, i] += 0.5 * ResultE[i, j] * (MuX[j, i] * Dx[i, 1] + 0.5 * VarX[j, i] * Dxx[i, 1])

            if i < numX - 1:
                U[j, i] += 0.5 * ResultE[i + 1, j] * (MuX[j, i] * Dx[i, 2] + 0.5 * VarX[j, i] * Dxx[i, 2])

    # explicit y
    for i in range(numX):
        for j in range(numY):

            V[i, j] = 0.0
            if j > 0:
                V[i, j] += ResultE[i, j - 1] * (MuY[i, j] * Dy[j, 0] + 0.5 * VarY[i, j] * Dyy[j, 0])
            V[i, j] += ResultE[i, j] * (MuY[i, j] * Dy[j, 1] + 0.5 * VarY[i, j] * Dyy[j, 1])
            if j < numY - 1:
                V[i, j] += ResultE[i, j + 1] * (MuY[i, j] * Dy[j, 2] + 0.5 * VarY[i, j] * Dyy[j, 2])

            U[j, i] += V[i, j]

    a = np.empty(numX, dtype=dace.float64)
    b = np.empty(numX, dtype=dace.float64)
    c = np.empty(numX, dtype=dace.float64)
    # implicit x
    # TODO DaCe: can this loop be a map? (likely yes, should be auto-optimized)
    for j in range(numY):

        a = -0.5 * (MuX[j, :] * Dx[:, 0] + 0.5 * VarX[j, :] * Dxx[:, 0])
        b = dtInv - 0.5 * (MuX[j, :] * Dx[:, 1] + 0.5 * VarX[j, :] * Dxx[:, 1])
        c = -0.5 * (MuX[j, :] * Dx[:, 2] + 0.5 * VarX[j, :] * Dxx[:, 2])
        # for i in range(numX):
        #     a[i] = -0.5 * (MuX[j, i] * Dx[i, 0] + 0.5 * VarX[j, i] * Dxx[i, 0])
        #     b[i] = dtInv - 0.5 * (MuX[j, i] * Dx[i, 1] + 0.5 * VarX[j, i] * Dxx[i, 1])
        #     c[i] = -0.5 * (MuX[j, i] * Dx[i, 2] + 0.5 * VarX[j, i] * Dxx[i, 2])

        uu = U[j, :numX]

        # tridag(numX, a, b, c, uu)
        dace_tridag(a, b, c, uu)

    aa = np.empty(numY, dtype=dace.float64)
    bb = np.empty(numY, dtype=dace.float64)
    cc = np.empty(numY, dtype=dace.float64)
    # implicit y
    # TODO DaCe: can this loop be a map? (likely yes, should be auto-optimized)
    for i in range(numX):
        aa = -0.5 * (MuY[i, :] * Dy[:, 0] + 0.5 * VarY[i, :] * Dyy[:, 0])
        bb = dtInv - 0.5 * (MuY[i, :] * Dy[:, 1] + 0.5 * VarY[i, :] * Dyy[:, 1])
        cc = -0.5 * (MuY[i, :] * Dy[:, 2] + 0.5 * VarY[i, :] * Dyy[:, 2])

        yy = ResultE[i, :numY]

        # TODO: DaCe: technically this loop can be rewritten, but then it does not validate if auto-opt is applied. It seems that
        # updates are not transferred back to ResultE (rather a transient array is created)
        # yy[:] = dtInv * U[:, i] - 0.5 * V[i, :]
        for j in range(numY):
            yy[j] = dtInv * U[j, i] - 0.5 * V[i, j]

        # )
        dace_tridag(aa, bb, cc, yy)


# alpha: float, nu: float, beta: float,
@dace.program
def main_loop(alpha: float, nu: float, beta: float, Time: dace.float64[numT], U: dace.float64[numY, numX],
              V: dace.float64[numX, numY], X: dace.float64[numX], Dx: dace.float64[numX, 3], Dxx: dace.float64[numX, 3],
              MuX: dace.float64[numY, numX], VarX: dace.float64[numY, numX], Y: dace.float64[numY],
              Dy: dace.float64[numY, 3], Dyy: dace.float64[numY, 3], MuY: dace.float64[numX, numY],
              VarY: dace.float64[numX, numY], ResultE: dace.float64[numX, numY]):

    for i in range(numT - 2, -1, -1):
        value = Time[i]
        Tvalue = Time[i]
        updateParams(i, alpha, beta, nu, X, Y, Time, MuX, VarX, MuY, VarY, value, Tvalue=Tvalue)

        rollback_dace(i, Time, U, V, Dx, Dxx, MuX, VarX, Dy, Dyy, MuY, VarY, ResultE)


N = 8
a = np.random.rand(N).astype(np.float64)
b = np.random.rand(N).astype(np.float64)
c = np.random.rand(N).astype(np.float64)
y = np.random.rand(N).astype(np.float64)

a_dace = np.copy(a)
b_dace = np.copy(b)
c_dace = np.copy(c)
y_dace = np.copy(y)

############ TRIDAG ALONE

# print(y)
# tridag(8, a, b, c, y)

# # tridag alone works (TODO: dace it)
# Ntridiag = N
# sdfg = dace_tridag.to_sdfg()
# sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.CPU)
# sdfg(a_dace, b_dace, c_dace, y_dace, Ntridiag=Ntridiag)
# assert np.allclose(y, y_dace)

g = 1
t = 5.0
alpha = 0.2
nu = 0.6
beta = 0.5

numX = 8
numY = 8
numZ = max(numX, numY)
numT = 4
a = np.random.rand(numZ).astype(np.float64)
b = np.random.rand(numZ).astype(np.float64)
c = np.random.rand(numZ).astype(np.float64)
y = np.random.rand(numZ).astype(np.float64)

a_dace = np.copy(a)
b_dace = np.copy(b)
c_dace = np.copy(c)
y_dace = np.copy(y)
Time = np.random.rand(numT).astype(np.float64)
U = np.random.rand(numY, numX).astype(np.float64)
V = np.random.rand(numX, numY).astype(np.float64)
Dx = np.random.rand(numX, 3).astype(np.float64)
Dxx = np.random.rand(numX, 3).astype(np.float64)
MuX = np.random.rand(numY, numX).astype(np.float64)
VarX = np.random.rand(numY, numX).astype(np.float64)
Dy = np.random.rand(numY, 3).astype(np.float64)
Dyy = np.random.rand(numY, 3).astype(np.float64)
MuY = np.random.rand(numX, numY).astype(np.float64)
VarY = np.random.rand(numX, numY).astype(np.float64)
ResultE = np.random.rand(numX, numY).astype(np.float64)
X = np.ndarray(numX).astype(np.float64)
Y = np.ndarray(numY).astype(np.float64)

Time_dace = np.copy(Time)
U_dace = np.copy(U)
V_dace = np.copy(V)
Dx_dace = np.copy(Dx)
Dxx_dace = np.copy(Dxx)
MuX_dace = np.copy(MuX)
VarX_dace = np.copy(VarX)
Dy_dace = np.copy(Dy)
Dyy_dace = np.copy(Dyy)
MuY_dace = np.copy(MuY)
VarY_dace = np.copy(VarY)
ResultE_dace = np.copy(ResultE)
X_dace = np.copy(X)
Y_dace = np.copy(Y)

############ ROLL BACK & TRIDAG

# rollback(g, a, b, c, Time, U, V, Dx, Dxx, MuX, VarX, Dy, Dyy, MuY, VarY, ResultE)

# rollback_sdfg = rollback_dace.to_sdfg()
# rollback_sdfg = auto_optimize(rollback_sdfg, dace.dtypes.DeviceType.CPU)
# rollback_sdfg(g,
#               Time_dace,
#               U_dace,
#               V_dace,
#               Dx_dace,
#               Dxx_dace,
#               MuX_dace,
#               VarX_dace,
#               Dy_dace,
#               Dyy_dace,
#               MuY_dace,
#               VarY_dace,
#               ResultE_dace,
#               numX=numX,
#               numT=numT,
#               numY=numY,
#               numZ=numZ)
# print(np.linalg.norm(ResultE - ResultE_dace) / np.linalg.norm(ResultE))
# assert np.allclose(ResultE, ResultE_dace)

############ Main Loop

main_loop_sdfg = main_loop.to_sdfg()
main_loop.f(alpha, nu, beta, Time, U, V, X, Dx, Dxx, MuX, VarX, Y, Dy, Dyy, MuY, VarY, ResultE)
main_loop_sdfg = auto_optimize(
    main_loop_sdfg,
    dace.dtypes.DeviceType.CPU)  # TODO: fix this: either fails because of free symbols or does not validate
main_loop_sdfg(alpha,
               nu,
               beta,
               Time_dace,
               U_dace,
               V_dace,
               X_dace,
               Dx_dace,
               Dxx_dace,
               MuX_dace,
               VarX_dace,
               Y_dace,
               Dy_dace,
               Dyy_dace,
               MuY_dace,
               VarY_dace,
               ResultE_dace,
               numX=numX,
               numY=numY,
               numT=numT,
               numZ=numZ)

print(np.linalg.norm(ResultE - ResultE_dace) / np.linalg.norm(ResultE))
assert np.allclose(ResultE, ResultE_dace)
