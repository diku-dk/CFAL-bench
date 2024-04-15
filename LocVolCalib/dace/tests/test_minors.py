"""

    Testbench for minor functions in VolCalib (initGrid, initOperator, setPayoff)    

    TODO: test for updateParam
"""

import dace

import utils
import numpy as np
import math
import time
from dace.transformation.auto.auto_optimize import auto_optimize
import copy

numX, numY, numT, N = (dace.symbol(s, dtype=dace.int32) for s in ('numX', 'numY', 'numT', 'N'))


@dace.program
def initGrid(s0: float, alpha: float, nu: float, t: float, X: dace.float64[numX], Y: dace.float64[numY],
             Time: dace.float64[numT]):

    # This works in DaCe
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


@dace.program
def initOperator(xx: dace.float64[N], D: dace.float64[N, 3], DD: dace.float64[N, 3], N=1):
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


@dace.program
def initOperator2(xx: dace.float64[N], D: dace.float64[N, 3], DD: dace.float64[N, 3]):
    """
    Initializes Globals Dx[Dy] and Dxx[Dyy]
    """

    # A more array-ish version

    # This works in DaCe
    dxls = 0.0
    dxus = xx[1] - xx[0]

    #lower boundary
    D[0, 0] = 0.0
    D[0, 1] = -1.0 / dxus
    D[0, 2] = 1.0 / dxus

    DD[0, 0] = 0.0
    DD[0, 1] = 0.0
    DD[0, 2] = 0.0

    dxl = np.empty((N - 2), dtype=dace.float64)
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
def setPayoff_base(strike: dace.float64, X: dace.float64[numX], ResultE: dace.float64[numX, numY]):

    # This is ok, auto-opt should be able to properly deal with this (creating two maps)
    for i in range(numX):
        payoff = max(X[i] - strike, 0.0)

        for j in range(numY):
            ResultE[i, j] = payoff


@dace.program
def setPayoff(strike: dace.float64, X: dace.float64[numX], ResultE: dace.float64[numX, numY]):
    # a more arrayish version
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
def updateParams_np(g: int, alpha: float, beta: float, nu: float, X: dace.float64[numX], Y: dace.float64[numY],
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
    # VarY.fill(nu**2)  # appar\ntly this is not ok
    for i in range(numX):
        for j in range(numY):
            MuY[i, j] = 0.0
            VarY[i, j] = nu * nu


outer, numX, numY, numT, s0, t, alpha, nu, beta = utils.getPrefedinedInputDataSet("xs")

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

# MuX = np.ndarray((numY, numX)).astype(np.float64)
# MuY = np.ndarray((numX, numY)).astype(np.float64)
# VarX = np.ndarray((numY, numX)).astype(np.float64)
# VarY = np.ndarray((numX, numY)).astype(np.float64)
# ResultE = np.ndarray((numX, numY)).astype(np.float64)

# result = np.ndarray(outer).astype(np.float64)

# Test Init Grid ######################

# X_ref = np.copy(X)
# Y_ref = np.copy(Y)
# Time_ref = np.copy(Time)

# indX_ref, indY_ref = initGrid.f(s0, alpha, nu, t, X_ref, Y_ref, Time_ref)
# sdfg = initGrid.to_sdfg()
# sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.CPU)
# indX, indY = sdfg(s0, alpha, nu, t, X, Y, Time, numX=numX, numY=numY, numT=numT)

# assert np.allclose(indX, indX_ref)
# assert np.allclose(indY, indY_ref)

################################

# Test Init Operator #####################

indX_ref, indY_ref = initGrid.f(s0, alpha, nu, t, X, Y, Time)
X_ref = np.copy(X)
Dx_ref = np.copy(Dx)
Dxx_ref = np.copy(Dxx)
a_ref = np.copy(a)
b_ref = np.copy(b)
c_ref = np.copy(c)

initOperator.f(X_ref, Dx_ref, Dxx_ref, N=numX)

# This is for the init of Y, not checked here against DaCe version
initOperator.f(Y, Dy, Dy, N=numY)

sdfg = initOperator2.to_sdfg()
sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.CPU)
sdfg(X, Dx, Dxx, N=numX)
print(np.linalg.norm(Dx - Dx_ref) / np.linalg.norm(Dx_ref))
assert np.allclose(Dxx, Dxx_ref)

# Test Set Payoff #####################

strike = np.float64(0.03)
ResultE = np.ndarray((numX, numY)).astype(np.float64)
ResultE_ref = np.copy(ResultE)

setPayoff_base.f(strike, X, ResultE_ref)

# TEST DACE
sdfg_set_payoff = setPayoff_base.to_sdfg()
sdfg_set_payoff = auto_optimize(sdfg_set_payoff, dace.dtypes.DeviceType.CPU)  # THIS one does not work!!
sdfg_set_payoff(strike, X, ResultE, numX=numX, numY=numY)
# assert np.allclose(ResultE, ResultE_ref)

assert np.allclose(ResultE, ResultE_ref)
##################
# TODO: test updateParams
