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
numX, numY, numT, N, Ntridiag, numZ, outer, __tmp112 = (dace.symbol(s, dtype=dace.int32)
                                                        for s in ('numX', 'numY', 'numT', 'N', 'Ntridiag', 'numZ',
                                                                  'outer', '__tmp112'))


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
    # payoff = np.maximum(X - strike, 0.0)
    # for i in range(numX):
    #     ResultE[i, :] = payoff[i]

    for i in range(numX):
        payoff = max(X[i] - strike, 0.0)

        for j in range(numY):
            ResultE[i, j] = payoff


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

    for j in range(numY):
        for i in range(numX):
            MuX[j, i] = 0.0
            VarX[j, i] = np.exp(2 * (beta * np.log(X[i]) + Y[j] - 0.5 * nu * nu * Time[g]))
            # VarX[j, i] = np.exp(2 * (beta * np.log(X[i]) + Y[j] - 0.5 * nu * nu * Tvalue))

    # MuY.fill(0)
    # VarY.fill(nu**2)  # apparantely currently not supported in DaCe is not ok
    for i in range(numX):
        for j in range(numY):
            MuY[i, j] = 0.0
            VarY[i, j] = nu * nu


@dace.program
def tridag(a: dace.float64[Ntridiag], b: dace.float64[Ntridiag], c: dace.float64[Ntridiag], y: dace.float64[Ntridiag]):
    """
    Computes the solution of the tridiagonal system in output array y
    - a contains the N-1 subdiagonal elements
    - b contains the N diagonal elements
    - c contains the N-1 superdiagonal elements
    - y right hand side and output
    """

    # TODO DaCe :
    # - rewrite in a more array-programminsh way (if possible)

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
def rollback(g: int, Time: dace.float64[numT], U: dace.float64[numY, numX], V: dace.float64[numX, numY],
             Dx: dace.float64[numX, 3], Dxx: dace.float64[numX, 3], MuX: dace.float64[numY, numX],
             VarX: dace.float64[numY, numX], Dy: dace.float64[numY, 3], Dyy: dace.float64[numY, 3],
             MuY: dace.float64[numX, numY], VarY: dace.float64[numX, numY], ResultE: dace.float64[numX, numY]):
    # DaCe:
    # - differently from the naive implementation, we allocate here a,b,c to favor symbol inference
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
        tridag(a, b, c, uu)

    aa = np.empty(numY, dtype=dace.float64)
    bb = np.empty(numY, dtype=dace.float64)
    cc = np.empty(numY, dtype=dace.float64)
    # implicit y
    # TODO DaCe: can this loop be a map? (likely yes, should be auto-optimized)
    for i in range(numX):
        aa = -0.5 * (MuY[i, :] * Dy[:, 0] + 0.5 * VarY[i, :] * Dyy[:, 0])
        bb = dtInv - 0.5 * (MuY[i, :] * Dy[:, 1] + 0.5 * VarY[i, :] * Dyy[:, 1])
        cc = -0.5 * (MuY[i, :] * Dy[:, 2] + 0.5 * VarY[i, :] * Dyy[:, 2])
        # for j in range(numY):
        #     aa[j] = -0.5 * (MuY[i, j] * Dy[j, 0] + 0.5 * VarY[i, j] * Dyy[j, 0])
        #     bb[j] = dtInv - 0.5 * (MuY[i, j] * Dy[j, 1] + 0.5 * VarY[i, j] * Dyy[j, 1])
        #     cc[j] = -0.5 * (MuY[i, j] * Dy[j, 2] + 0.5 * VarY[i, j] * Dyy[j, 2])

        yy = ResultE[i, :numY]

        # TODO: DaCe: technically this loop can be rewritten, but then it does not validate if auto-opt is applied. It seems that
        # updates are not transferred back to ResultE (rather a transient array is created)
        # yy[:] = dtInv * U[:, i] - 0.5 * V[i, :]
        for j in range(numY):
            yy[j] = dtInv * U[j, i] - 0.5 * V[i, j]

        # tridag(
        #     numY,
        #     a,
        #     b,
        #     c,
        #     yy,
        # )
        tridag(aa, bb, cc, yy)


# def value(s0: float, strike: float, t: float, alpha: float, nu: float, beta: float, numX: int, numY: int, numT: int, a,
#           b, c, Time, U, V, X, Dx, Dxx, MuX, VarX, Y, Dy, Dyy, MuY, VarY, ResultE):
@dace.program
def value(s0: float, strike: float, t: float, alpha: float, nu: float, beta: float, Time: dace.float64[numT],
          U: dace.float64[numY, numX], V: dace.float64[numX, numY], X: dace.float64[numX], Dx: dace.float64[numX, 3],
          Dxx: dace.float64[numX, 3], MuX: dace.float64[numY, numX], VarX: dace.float64[numY, numX],
          Y: dace.float64[numY], Dy: dace.float64[numY, 3], Dyy: dace.float64[numY, 3], MuY: dace.float64[numX, numY],
          VarY: dace.float64[numX, numY], ResultE: dace.float64[numX, numY]):

    # indX, indY = initGrid(numX, numY, numT, s0, alpha, nu, t, X, Y, Time)
    # indX, indY = initGrid(s0, alpha, nu, t, X, Y, Time, numX=numX, numY=numY, numT=numT)
    indX, indY = initGrid(s0, alpha, nu, t, X, Y, Time)

    # initOperator(numX, X, Dx, Dxx)
    # initOperator(numY, Y, Dy, Dyy)

    initOperator(X, Dx, Dxx)
    initOperator(Y, Dy, Dyy)

    setPayoff(strike, X, ResultE)

    for i in range(numT - 2, -1, -1):

        updateParams(i, alpha, beta, nu, X, Y, Time, MuX, VarX, MuY, VarY)
        rollback(i, Time, U, V, Dx, Dxx, MuX, VarX, Dy, Dyy, MuY, VarY, ResultE)

    res = ResultE[indX, indY]
    return res


@dace.program
def VolCalib(s0: float, t: float, alpha: float, nu: float, beta: float, Time: dace.float64[numT],
             U: dace.float64[numY, numX], V: dace.float64[numX, numY], X: dace.float64[numX], Dx: dace.float64[numX, 3],
             Dxx: dace.float64[numX, 3], MuX: dace.float64[numY, numX], VarX: dace.float64[numY, numX],
             Y: dace.float64[numY], Dy: dace.float64[numY, 3], Dyy: dace.float64[numY, 3], MuY: dace.float64[numX,
                                                                                                             numY],
             VarY: dace.float64[numX, numY], ResultE: dace.float64[numX, numY], result: dace.float64[outer]):

    # tmp = np.empty(outer, dtype=dace.float64)
    for i in range(outer):
        strike = 0.001 * i
        # compute
        result[i] = value(s0, strike, t, alpha, nu, beta, Time, U, V, X, Dx, Dxx, MuX, VarX, Y, Dy, Dyy, MuY, VarY,
                          ResultE)
    # return tmp


#main
dataset_size = "XS"
outer, numX, numY, numT, s0, t, alpha, nu, beta = utils.getPrefedinedInputDataSet(dataset_size)

#global array allocation
numZ = max(numX, numY)
# a = np.ndarray(numZ).astype(np.float64)
# b = np.ndarray(numZ).astype(np.float64)
# c = np.ndarray(numZ).astype(np.float64)
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

VolCalib_sdfg = VolCalib.to_sdfg()
# VolCalib_sdfg = auto_optimize(VolCalib_sdfg, dace.dtypes.DeviceType.CPU)  # THis one does not work

VolCalib_sdfg(s0,
              t,
              alpha,
              nu,
              beta,
              Time,
              U,
              V,
              X,
              Dx,
              Dxx,
              MuX,
              VarX,
              Y,
              Dy,
              Dyy,
              MuY,
              VarY,
              ResultE,
              result,
              numX=numX,
              numT=numT,
              numY=numY,
              numZ=numZ,
              outer=outer,
              __tmp112=0)

end = time.time()

print(f"Time in usecs: {(end-start)*1e6}")

# validate (currently also printing out actual results for inspection)
print("Computed results: ", result)
print("------------")
print("Expected results: ", utils.getPrefefinedOutputDataSet(dataset_size))
print("------------")
print(
    print(
        np.linalg.norm(result - utils.getPrefefinedOutputDataSet(dataset_size)) /
        np.linalg.norm(utils.getPrefefinedOutputDataSet(dataset_size))))
assert np.allclose(result, utils.getPrefefinedOutputDataSet(dataset_size))