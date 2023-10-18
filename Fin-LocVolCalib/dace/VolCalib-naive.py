"""

    Naive python implementation of the VolCalib benchmark
"""
import utils
import numpy as np
import math
import time


def initGrid(numX: int, numY: int, numT: int, s0: float, alpha: float, nu: float, t: float, X, Y, Time):
    """
    Initializes indX, indY, X, Y, and Time
    """

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


def initOperator(N: int, xx, D, DD):
    """
    Initializes Globals Dx[Dy] and Dxx[Dyy]
    """

    dxl = 0.0
    dxu = xx[1] - xx[0]

    #lower boundary
    D[0, 0] = 0.0
    D[0, 1] = -1.0 / dxu
    D[0, 2] = 1.0 / dxu

    DD[0, 0] = 0.0
    DD[0, 1] = 0.0
    DD[0, 2] = 0.0

    #inner
    for i in range(1, N - 1):
        dxl = xx[i] - xx[i - 1]
        dxu = xx[i + 1] - xx[i]

        D[i, 0] = -dxu / dxl / (dxl + dxu)
        D[i, 1] = (dxu / dxl - dxl / dxu) / (dxl + dxu)
        D[i, 2] = dxl / dxu / (dxl + dxu)

        DD[i, 0] = 2.0 / dxl / (dxl + dxu)
        DD[i, 1] = -2.0 * (1.0 / dxl + 1.0 / dxu) / (dxl + dxu)
        DD[i, 2] = 2.0 / dxu / (dxl + dxu)

    #	upper boundary
    dxl = xx[N - 1] - xx[N - 2]
    dxu = 0.0

    D[(N - 1), 0] = -1.0 / dxl
    D[(N - 1), 1] = 1.0 / dxl
    D[(N - 1), 2] = 0.0

    DD[(N - 1), 0] = 0.0
    DD[(N - 1), 1] = 0.0
    DD[(N - 1), 2] = 0.0


def setPayoff(numX: int, numY: int, strike: float, X, ResultE):

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


def tridag(N: int, a, b, c, y):
    """
    Computes the solution of the tridiagonal system in output array y
    - a contains the N-1 subdiagonal elements
    - b contains the N diagonal elements
    - c contains the N-1 superdiagonal elements
    - y right hand side and output
    """

    #forward swap
    for i in range(1, N - 1):
        beta = a[i] / b[i - 1]

        b[i] = b[i] - beta * c[i - 1]
        y[i] = y[i] - beta * y[i - 1]

    #backward
    y[N - 1] = y[N - 1] / b[N - 1]
    for i in range(N - 2, -1, -1):
        y[i] = (y[i] - c[i] * y[i + 1]) / b[i]


def rollback(numX: int, numY: int, g: int, a, b, c, Time, U, V, Dx, Dxx, MuX, VarX, Dy, Dyy, MuY, VarY, ResultE):

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

    # implicit x
    for j in range(numY):
        for i in range(numX):
            a[i] = -0.5 * (MuX[j, i] * Dx[i, 0] + 0.5 * VarX[j, i] * Dxx[i, 0])
            b[i] = dtInv - 0.5 * (MuX[j, i] * Dx[i, 1] + 0.5 * VarX[j, i] * Dxx[i, 1])
            c[i] = -0.5 * (MuX[j, i] * Dx[i, 2] + 0.5 * VarX[j, i] * Dxx[i, 2])

        uu = U[j, :]

        tridag(numX, a, b, c, uu)

    # implicit y
    for i in range(numX):
        for j in range(numY):
            a[j] = -0.5 * (MuY[i, j] * Dy[j, 0] + 0.5 * VarY[i, j] * Dyy[j, 0])
            b[j] = dtInv - 0.5 * (MuY[i, j] * Dy[j, 1] + 0.5 * VarY[i, j] * Dyy[j, 1])
            c[j] = -0.5 * (MuY[i, j] * Dy[j, 2] + 0.5 * VarY[i, j] * Dyy[j, 2])

        yy = ResultE[i, :]

        for j in range(numY):
            yy[j] = dtInv * U[j, i] - 0.5 * V[i, j]

        tridag(numY, a, b, c, yy)


def value(s0: float, strike: float, t: float, alpha: float, nu: float, beta: float, numX: int, numY: int, numT: int, a,
          b, c, Time, U, V, X, Dx, Dxx, MuX, VarX, Y, Dy, Dyy, MuY, VarY, ResultE):

    indX, indY = initGrid(numX, numY, numT, s0, alpha, nu, t, X, Y, Time)
    initOperator(numX, X, Dx, Dxx)
    initOperator(numY, Y, Dy, Dyy)

    setPayoff(numX, numY, strike, X, ResultE)

    for i in range(numT - 2, -1, -1):

        updateParams(numX, numY, i, alpha, beta, nu, X, Y, Time, MuX, VarX, MuY, VarY)

        rollback(numX, numY, i, a, b, c, Time, U, V, Dx, Dxx, MuX, VarX, Dy, Dyy, MuY, VarY, ResultE)

    res = ResultE[indX, indY]
    return res


#main
dataset_size = "s"
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
print(result)
print("------------")
print(utils.getPrefefinedOutputDataSet(dataset_size))
print("------------")
print(
    print(
        np.linalg.norm(result - utils.getPrefefinedOutputDataSet(dataset_size)) /
        np.linalg.norm(utils.getPrefefinedOutputDataSet(dataset_size))))
assert np.allclose(result, utils.getPrefefinedOutputDataSet(dataset_size))
# TODO