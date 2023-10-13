"""

    DaCe implementation of the VolCalib benchmark
"""
import utils
import numpy as np
import math
import time

import dace
from dace.transformation.auto.auto_optimize import auto_optimize

# TODOs:
# - promote also s0, alpha, nu, ... to symbols?
numX, numY, numT, N, Ntridiag, numZ = (dace.symbol(s, dtype=dace.int32)
                                       for s in ('numX', 'numY', 'numT', 'N', 'Ntridiag', 'numZ'))


@dace.program
def initGrid(s0: float, alpha: float, nu: float, t: float, X: dace.float64[numX], Y: dace.float64[numY],
             Time: dace.float64[numT]):
    """
    Initializes indX, indY, X, Y, and Time
    """

    # DaCe: auto-optimize should easily be able to transform these loops in maps
    for i in range(numT):
        Time[i] = t * i / (numT - 1)

    stdX = 20 * alpha * s0 * math.sqrt(t)
    dx = stdX / numX
    indX = int(s0 / dx)

    for i in range(numX):
        X[i] = i * dx - indX * dx + s0

    stdY = 10 * nu * math.sqrt(t)
    dy = stdY / numY
    logAlpha = math.log(alpha)
    indY = int(numY / 2)

    for i in range(numY):
        Y[i] = i * dy - indY * dy + logAlpha
    return indX, indY


@dace.program
def initOperator(xx: dace.float64[N], D: dace.float64[N, 3], DD: dace.float64[N, 3]):
    """
    Initializes Globals Dx[Dy] and Dxx[Dyy]
    """
    dxls = 0.0
    dxus = xx[1] - xx[0]

    #lower boundary
    D[0, 0] = 0.0
    D[0, 1] = -1.0 / dxus
    D[0, 2] = 1.0 / dxus

    DD[0, 0] = 0.0
    DD[0, 1] = 0.0
    DD[0, 2] = 0.0

    # array version of the main loop in the function
    dxl = xx[1:-1] - xx[0:-2]
    dxu = xx[2:] - xx[1:-1]

    D[1:-1, 0] = -dxu / dxl / (dxl + dxu)
    D[1:-1, 1] = (dxu / dxl - dxl / dxu) / (dxl + dxu)
    D[1:-1, 2] = dxl / dxu / (dxl + dxu)

    DD[1:-1, 0] = 2.0 / dxl / (dxl + dxu)
    DD[1:-1, 1] = -2.0 * (1.0 / dxl + 1.0 / dxu) / (dxl + dxu)
    DD[1:-1, 2] = 2.0 / dxu / (dxl + dxu)

    #	upper boundary
    dxls = xx[N - 1] - xx[N - 2]
    dxus = 0.0

    D[(N - 1), 0] = -1.0 / dxls
    D[(N - 1), 1] = 1.0 / dxls
    D[(N - 1), 2] = 0.0

    DD[(N - 1), 0] = 0.0
    DD[(N - 1), 1] = 0.0
    DD[(N - 1), 2] = 0.0


@dace.program
def setPayoff(strike: dace.float64, X: dace.float64[numX], ResultE: dace.float64[numX, numY]):
    payoff = np.maximum(X - strike, 0.0)
    for i in range(numX):
        ResultE[i, :] = payoff[i]


def updateParams(numX: int, numY: int, g: int, alpha: float, beta: float, nu: float, X, Y, Time, MuX, VarX, MuY, VarY):

    for j in range(numY):
        for i in range(numX):
            MuX[j, i] = 0.0
            VarX[j, i] = np.exp(2 * (beta * np.log(X[i]) + Y[j] - 0.5 * nu * nu * Time[g]))

    for i in range(numX):
        for j in range(numY):
            MuY[i, j] = 0.0
            VarY[i, j] = nu * nu


@dace.program
def updateParams(g: int, alpha: float, beta: float, nu: float, X: dace.float64[numX], Y: dace.float64[numY],
                 Time: dace.float64[numT], MuX: dace.float64[numY, numX], VarX: dace.float64[numY, numX],
                 MuY: dace.float64[numX, numY], VarY: dace.float64[numX, numY]):

    # DaCE
    # Maybe can be improved for readability/functional-friendliness
    # - second loop nest can be avoided but currently we are missing fill support
    # - the first one maybe can be further simplified?

    MuX.fill(0)

    for j in range(numY):

        VarX[j, :] = np.exp(2 * (beta * np.log(X) + Y[j] - 0.5 * nu * nu * Time[g]))
        # for i in range(numX):
        #     # MuX[j, i] = 0.0
        #     value = (beta * np.log(X[i]) + Y[j] - 0.5 * nu * nu * Time[g])
        #     VarX[j, i] = np.exp(2 * value)

    # MuY.fill(0)
    # VarY.fill(nu**2)  # apparantely currently not supported in DaCe is not ok
    for i in range(numX):
        for j in range(numY):
            MuY[i, j] = 0.0
            VarY[i, j] = nu * nu


# @dace.program
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
        # tridag(a, b, c, yy, Ntridiag=numY)


def value(s0: float, strike: float, t: float, alpha: float, nu: float, beta: float, numX: int, numY: int, numT: int, a,
          b, c, Time, U, V, X, Dx, Dxx, MuX, VarX, Y, Dy, Dyy, MuY, VarY, ResultE):

    # TODO: have this function as DaCe Program itself
    # indX, indY = initGrid(numX, numY, numT, s0, alpha, nu, t, X, Y, Time)
    initGrid_sdfg = initGrid.to_sdfg()
    initGrid_sdfg = auto_optimize(initGrid_sdfg, dace.dtypes.DeviceType.CPU)
    indX, indY = initGrid_sdfg(s0, alpha, nu, t, X, Y, Time, numX=numX, numY=numY, numT=numT)

    # initOperator(numX, X, Dx, Dxx)
    # initOperator(numY, Y, Dy, Dyy)

    initOperator_sdfg = initOperator.to_sdfg()
    initOperator_sdfg = auto_optimize(initOperator_sdfg, dace.dtypes.DeviceType.CPU)
    initOperator_sdfg(X, Dx, Dxx, N=numX)
    initOperator_sdfg(Y, Dy, Dyy, N=numY)

    # setPayoff(numX, numY, strike, X, ResultE)
    setPayoff_sdfg = setPayoff.to_sdfg()
    # setPayoff_sdfg = auto_optimize(setPayoff_sdfg, dace.dtypes.DeviceType.CPU) # THIS one does not work!!
    setPayoff_sdfg(strike, X, ResultE, numX=numX, numY=numY)

    updateParams_sdfg = updateParams.to_sdfg()
    updateParams_sdfg = auto_optimize(updateParams_sdfg, dace.dtypes.DeviceType.CPU)

    # rollback_sdfg = rollback.to_sdfg()
    for i in range(numT - 2, -1, -1):

        # updateParams(numX, numY, i, alpha, beta, nu, X, Y, Time, MuX, VarX, MuY, VarY)

        updateParams_sdfg(i, alpha, beta, nu, X, Y, Time, MuX, VarX, MuY, VarY, numX=numX, numY=numY, numT=numT)
        rollback.f(i, a, b, c, Time, U, V, Dx, Dxx, MuX, VarX, Dy, Dyy, MuY, VarY, ResultE)
        # rollback.f(i,
        #            a,
        #            b,
        #            c,
        #            Time,
        #            U,
        #            V,
        #            Dx,
        #            Dxx,
        #            MuX,
        #            VarX,
        #            Dy,
        #            Dyy,
        #            MuY,
        #            VarY,
        #            ResultE,
        #            numX=numX,
        #            numY=numY,
        #            numZ=max(numX, numY))

    res = ResultE[indX, indY]
    return res


#main
dataset_size = "xs"
outer, numX, numY, numT, s0, t, alpha, nu, beta = utils.getPrefedinedInputDataSet(dataset_size)

#global array allocation
num_z = max(numX, numY)
a = np.ndarray(num_z).astype(np.float64)
b = np.ndarray(num_z).astype(np.float64)
c = np.ndarray(num_z).astype(np.float64)
V = np.ndarray((numX, numY)).astype(np.float64)
U = np.ndarray((numY, numX)).astype(np.float64)

X = np.ndarray(numX).astype(np.float64)
Dx = np.ndarray((numX, 3)).astype(np.float64)
Dxx = np.ndarray((numX, 3)).astype(np.float64)
Y = np.ndarray(numY).astype(np.float64)
Dy = np.ndarray((numY, 3)).astype(np.float64)
Dyy = np.ndarray((numY, 3)).astype(np.float64)
Time = np.ndarray(numT).astype(np.float64)

MuX = np.ndarray((numY, numX)).astype(np.float64)
MuY = np.ndarray((numX, numY)).astype(np.float64)
VarX = np.ndarray((numY, numX)).astype(np.float64)
VarY = np.ndarray((numX, numY)).astype(np.float64)
ResultE = np.ndarray((numX, numY)).astype(np.float64)

result = np.ndarray(outer).astype(np.float64)

### main computation kernel

start = time.time()
for i in range(outer):
    strike = 0.001 * i
    # compute
    result[i] = value(s0, strike, t, alpha, nu, beta, numX, numY, numT, a, b, c, Time, U, V, X, Dx, Dxx, MuX, VarX, Y,
                      Dy, Dyy, MuY, VarY, ResultE)
end = time.time()

print(f"Time in usecs: {(end-start)*1e6}")
# validate

assert np.allclose(result, utils.getPrefefinedOutputDataSet(dataset_size))
print(result)
# TODO