import copy
import dace
import numpy as np
import numpy.typing as npt
import time
import utils


from dace.transformation.auto.auto_optimize import apply_gpu_storage, auto_optimize


outer, numXY, numX, numY, numT = dace.symbol('outer'), dace.symbol('numXY'), dace.symbol('numX'), dace.symbol('numY'), dace.symbol('numT')
M, N = (dace.symbol(s) for s in ('M', 'N'))
g, gidx = dace.symbol('g'), dace.symbol('gidx')
xidx, yidx = dace.symbol('xidx'), dace.symbol('yidx')
# indX, indY = dace.symbol('indX'), dace.symbol('indY')


@dace.program
def initGrid_dace(s0: float, alpha: float, nu: float, t: float,
                  X: dace.float64[numX],
                  Y: dace.float64[numY],
                  Time: dace.float64[numT]):
    """ Initializes indX, indY, X, Y, and Time. """

    # Scalar computations
    # indX
    stdX = 20 * alpha * s0 * np.sqrt(t)
    dx = stdX / numX
    indX = int(s0 / dx)
    # indY
    stdY = 10 * nu * np.sqrt(t)
    dy = stdY / numY
    logAlpha = np.log(alpha)
    indY = int(numY / 2)

    # Array computations
    # Time
    for i in dace.map[0:numT]:
        Time[i] = t * i / (numT - 1)
    # X
    for i in dace.map[0:numX]:
        X[i] = i * dx - indX * dx + s0
    # Y
    for i in dace.map[0:numY]:
        Y[i] = i * dy - indY * dy + logAlpha

    return indX, indY


@dace.program
def initOperator_dace(xx: dace.float64[M], D: dace.float64[M, 3], DD: dace.float64[M,3]):
    """ Initializes Globals Dx[Dy] and Dxx[Dyy]. """

    dxll = 0.0
    dxul = xx[1] - xx[0]

    # lower boundary
    D[0, 0] = 0.0
    D[0, 1] = -1.0 / dxul
    D[0, 2] = 1.0 / dxul

    DD[0, 0] = 0.0
    DD[0, 1] = 0.0
    DD[0, 2] = 0.0

    #inner
    for i in dace.map[1:M - 1]:
        dxl = xx[i] - xx[i - 1]
        dxu = xx[i + 1] - xx[i]

        D[i, 0] = -dxu / dxl / (dxl + dxu)
        D[i, 1] = (dxu / dxl - dxl / dxu) / (dxl + dxu)
        D[i, 2] = dxl / dxu / (dxl + dxu)

        DD[i, 0] = 2.0 / dxl / (dxl + dxu)
        DD[i, 1] = -2.0 * (1.0 / dxl + 1.0 / dxu) / (dxl + dxu)
        DD[i, 2] = 2.0 / dxu / (dxl + dxu)

    # upper boundary
    dxlu = xx[M - 1] - xx[M - 2]
    dxuu = 0.0

    D[(M - 1), 0] = -1.0 / dxlu
    D[(M - 1), 1] = 1.0 / dxlu
    D[(M - 1), 2] = 0.0

    DD[(M - 1), 0] = 0.0
    DD[(M - 1), 1] = 0.0
    DD[(M - 1), 2] = 0.0


@dace.program
def setPayoff_dace(strike: float, X: dace.float64[numX], ResultE: dace.float64[numX, numY]):

    for ip in dace.map[0:numX]:
        ResultE[ip] = max(X[ip] - strike, 0.0)
    # ResultE[:] = np.reshape(np.maximum(X - strike, 0.0), (numX, 1))


@dace.program
def updateParams_dace(alpha: float, beta: float, nu: float,
                      X: dace.float64[numX], Y: dace.float64[numY], Time: dace.float64[numT],
                      MuX: dace.float64[numY, numX], VarX: dace.float64[numY, numX],
                      MuY: dace.float64[numX, numY], VarY: dace.float64[numX, numY]):

    for ju in dace.map[0:numY]:
        for iu in dace.map[0:numX]:
            # MuX[ju, iu] = 0.0
            VarX[ju, iu] = np.exp(2 * (beta * np.log(X[iu]) + Y[ju] - 0.5 * nu * nu * Time[g]))
            # MuY[iu, ju] = 0.0
    # MuX[:] = 0.0
    # VarX[:] = np.exp(2 * np.add.outer(Y - 0.5 * nu * nu * Time[g], beta * np.log(X)))

    # MuY[:] = 0.0
    VarY[:] = nu * nu


@dace.program
def tridiag_dace(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], y: dace.float64[N]):
    """
    Computes the solution of the tridiagonal system in output array y
    - a contains the N-1 subdiagonal elements
    - b contains the N diagonal elements
    - c contains the N-1 superdiagonal elements
    - y right hand side and output
    """

    # forward swap
    for i in range(1, N - 1):
        beta = a[i] / b[i - 1]

        b[i] = b[i] - beta * c[i - 1]
        y[i] = y[i] - beta * y[i - 1]

    # backward
    y[N - 1] = y[N - 1] / b[N - 1]
    for i in range(N - 2, -1, -1):
        y[i] = (y[i] - c[i] * y[i + 1]) / b[i]


# @dace.program
# def rollback_dace(a: dace.float64[numXY, numXY], b: dace.float64[numXY, numXY], c: dace.float64[numXY, numXY],
#                   Time: dace.float64[numT], U: dace.float64[numY, numX], V: dace.float64[numX, numY],
#                   Dx: dace.float64[numX, 3], Dxx: dace.float64[numX, 3],
#                   MuX: dace.float64[numY, numX], VarX: dace.float64[numY, numX],
#                   Dy: dace.float64[numY, 3], Dyy: dace.float64[numY, 3],
#                   MuY: dace.float64[numX, numY], VarY: dace.float64[numX, numY],
#                   ResultE: dace.float64[numX, numY]):

#     dtInv = 1.0 / (Time[gidx + 1] - Time[gidx])

#     # explicit x
#     for j, i in dace.map[0:numY, 0:numX]:
#         U[j, i] = dtInv * ResultE[i, j] + 0.5 * ResultE[i, j] * (MuX[j, i] * Dx[i, 1] + 0.5 * VarX[j, i] * Dxx[i, 1])
#         if i > 0:
#             U[j, i] = U[j, i] + 0.5 * ResultE[i - 1, j] * (MuX[j, i] * Dx[i, 0] + 0.5 * VarX[j, i] * Dxx[i, 0])
#         if i < numX - 1:
#             U[j, i] = U[j, i] + 0.5 * ResultE[i + 1, j] * (MuX[j, i] * Dx[i, 2] + 0.5 * VarX[j, i] * Dxx[i, 2])

#     # explicit y
#     for i, j in dace.map[0:numX, 0:numY]:
#         V[i, j] = ResultE[i, j] * (MuY[i, j] * Dy[j, 1] + 0.5 * VarY[i, j] * Dyy[j, 1])
#         if j > 0:
#             V[i, j] = V[i, j] + ResultE[i, j - 1] * (MuY[i, j] * Dy[j, 0] + 0.5 * VarY[i, j] * Dyy[j, 0])
#         if j < numY - 1:
#             V[i, j] = V[i, j] + ResultE[i, j + 1] * (MuY[i, j] * Dy[j, 2] + 0.5 * VarY[i, j] * Dyy[j, 2])
#         U[j, i] = U[j, i] + V[i, j]

#     # implicit x
#     for j in dace.map[0:numY]:
#         for i in range(numX):
#             a[j, i] = -0.5 * (MuX[j, i] * Dx[i, 0] + 0.5 * VarX[j, i] * Dxx[i, 0])
#             b[j, i] = dtInv - 0.5 * (MuX[j, i] * Dx[i, 1] + 0.5 * VarX[j, i] * Dxx[i, 1])
#             c[j, i] = -0.5 * (MuX[j, i] * Dx[i, 2] + 0.5 * VarX[j, i] * Dxx[i, 2])

#         # uu = U[j, :]

#         tridiag_dace(a[j, :numX], b[j, :numX], c[j, :numX], U[j, :])

#     # implicit y
#     for i in dace.map[0:numX]:
#         for j in range(numY):
#             a[i, j] = -0.5 * (MuY[i, j] * Dy[j, 0] + 0.5 * VarY[i, j] * Dyy[j, 0])
#             b[i, j] = dtInv - 0.5 * (MuY[i, j] * Dy[j, 1] + 0.5 * VarY[i, j] * Dyy[j, 1])
#             c[i, j] = -0.5 * (MuY[i, j] * Dy[j, 2] + 0.5 * VarY[i, j] * Dyy[j, 2])
#             ResultE[i, j] = dtInv * U[j, i] - 0.5 * V[i, j]

#         # yy = ResultE[i, :]

#         # for j in range(numY):
#         #     yy[j] = dtInv * U[j, i] - 0.5 * V[i, j]

#         tridiag_dace(a[i, :numY], b[i, :numY], c[i, :numY], ResultE[i, :])


@dace.program
def rollback_dace(a: dace.float64[numXY, numXY], b: dace.float64[numXY, numXY], c: dace.float64[numXY, numXY],
                  Time: dace.float64[numT], U: dace.float64[numY, numX], V: dace.float64[numX, numY],
                  Dx: dace.float64[numX, 3], Dxx: dace.float64[numX, 3],
                  MuX: dace.float64[numY, numX], VarX: dace.float64[numY, numX],
                  Dy: dace.float64[numY, 3], Dyy: dace.float64[numY, 3],
                  MuY: dace.float64[numX, numY], VarY: dace.float64[numX, numY],
                  ResultE: dace.float64[numX, numY]):

    dtInv = 1.0 / (Time[gidx + 1] - Time[gidx])

    # explicit x
    for j, i in dace.map[0:numY, 0:numX]:
        U[j, i] = dtInv * ResultE[i, j] + 0.25 * ResultE[i, j] * VarX[j, i] * Dxx[i, 1]
        if i > 0:
            U[j, i] = U[j, i] + 0.25 * ResultE[i - 1, j] * VarX[j, i] * Dxx[i, 0]
        if i < numX - 1:
            U[j, i] = U[j, i] + 0.25 * ResultE[i + 1, j] * VarX[j, i] * Dxx[i, 2]

    # explicit y
    # for i, j in dace.map[0:numX, 0:numY]:
    for j, i in dace.map[0:numY, 0:numX]:
        V[i, j] = 0.5 * ResultE[i, j] * VarY[i, j] * Dyy[j, 1]
        if j > 0:
            V[i, j] = V[i, j] + ResultE[i, j - 1] * 0.5 * VarY[i, j] * Dyy[j, 0]
        if j < numY - 1:
            V[i, j] = V[i, j] + ResultE[i, j + 1] * 0.5 * VarY[i, j] * Dyy[j, 2]
        U[j, i] = U[j, i] + V[i, j]

    # implicit x
    for j in dace.map[0:numY]:
        for i in range(numX):
            a[j, i] = -0.25 * VarX[j, i] * Dxx[i, 0]
            b[j, i] = dtInv - 0.25 * VarX[j, i] * Dxx[i, 1]
            c[j, i] = -0.25 * VarX[j, i] * Dxx[i, 2]

        # uu = U[j, :]

        tridiag_dace(a[j, :numX], b[j, :numX], c[j, :numX], U[j, :])

    # implicit y
    for i in dace.map[0:numX]:
        for j in range(numY):
            a[i, j] = -0.25 * VarY[i, j] * Dyy[j, 0]
            b[i, j] = dtInv - 0.25 * VarY[i, j] * Dyy[j, 1]
            c[i, j] = -0.25 * VarY[i, j] * Dyy[j, 2]
            ResultE[i, j] = dtInv * U[j, i] - 0.5 * V[i, j]

        # yy = ResultE[i, :]

        # for j in range(numY):
        #     yy[j] = dtInv * U[j, i] - 0.5 * V[i, j]

        tridiag_dace(a[i, :numY], b[i, :numY], c[i, :numY], ResultE[i, :])


@dace.program
def value_dace(strike: float, alpha: float, beta: float, nu: float,
               a: dace.float64[numXY, numXY], b: dace.float64[numXY, numXY], c: dace.float64[numXY, numXY],
               X: dace.float64[numX], Y: dace.float64[numY], Time: dace.float64[numT],
               U: dace.float64[numY, numX], V: dace.float64[numX, numY],
               Dx: dace.float64[numX, 3], Dxx: dace.float64[numX, 3],
               MuX: dace.float64[numY, numX], VarX: dace.float64[numY, numX],
               Dy: dace.float64[numY, 3], Dyy: dace.float64[numY, 3],
               MuY: dace.float64[numX, numY], VarY: dace.float64[numX, numY],
               ResultE: dace.float64[numX, numY]):
    
    setPayoff_dace(strike, X, ResultE)
    for it in range(numT - 2, -1, -1):
        updateParams_dace(alpha, beta, nu, X, Y, Time, MuX, VarX, MuY, VarY, g=it)
        rollback_dace(a, b, c, Time, U, V, Dx, Dxx, MuX, VarX, Dy, Dyy, MuY, VarY, ResultE, gidx=it)
    
    res = ResultE[xidx, yidx]
    return res


# @dace.program
# def LocVolCalib_dace(s0: float, alpha: float, beta: float, nu: float, t: float,
#                      a: dace.float64[outer, numXY, numXY], b: dace.float64[outer, numXY, numXY], c: dace.float64[outer, numXY, numXY],
#                      X: dace.float64[numX], Y: dace.float64[numY], Time: dace.float64[numT],
#                      U: dace.float64[outer, numY, numX], V: dace.float64[outer, numX, numY],
#                      Dx: dace.float64[numX, 3], Dxx: dace.float64[numX, 3],
#                      MuX: dace.float64[outer, numY, numX], VarX: dace.float64[outer, numY, numX],
#                      Dy: dace.float64[numY, 3], Dyy: dace.float64[numY, 3],
#                      MuY: dace.float64[outer, numX, numY], VarY: dace.float64[outer, numX, numY],
#                      ResultE: dace.float64[outer, numX, numY], result: dace.float64[outer]):
    
#     indX, indY = initGrid_dace(s0, alpha, nu, t, X, Y, Time)
#     initOperator_dace(X, Dx, Dxx)
#     initOperator_dace(Y, Dy, Dyy)
#     for io in dace.map[0:outer]:
#         strike = 0.001 * dace.float64(io)
#         result[io] = value_dace(strike, alpha, beta, nu, a[io], b[io], c[io], X, Y, Time,
#                                 U[io], V[io], Dx, Dxx, MuX[io], VarX[io], Dy, Dyy, MuY[io], VarY[io],
#                                 ResultE[io], xidx=indX, yidx=indY)


@dace.program
def LocVolCalib_dace(s0: float, alpha: float, beta: float, nu: float, t: float,
                     a: dace.float64[outer, numXY, numXY], b: dace.float64[outer, numXY, numXY], c: dace.float64[outer, numXY, numXY],
                     X: dace.float64[numX], Y: dace.float64[numY], Time: dace.float64[numT],
                     U: dace.float64[outer, numY, numX], V: dace.float64[outer, numX, numY],
                     Dx: dace.float64[numX, 3], Dxx: dace.float64[numX, 3],
                     MuX: dace.float64[outer, numY, numX], VarX: dace.float64[outer, numY, numX],
                     Dy: dace.float64[numY, 3], Dyy: dace.float64[numY, 3],
                     MuY: dace.float64[outer, numX, numY], VarY: dace.float64[outer, numX, numY],
                     ResultE: dace.float64[outer, numX, numY], result: dace.float64[outer]):
    
    indX, indY = initGrid_dace(s0, alpha, nu, t, X, Y, Time)
    initOperator_dace(X, Dx, Dxx)
    initOperator_dace(Y, Dy, Dyy)

    for io in dace.map[0:outer]:
        strike = 0.001 * dace.float64(io)
        setPayoff_dace(strike, X, ResultE[io])

    for it in range(numT - 2, -1, -1):
        for io in dace.map[0:outer]:
            updateParams_dace(alpha, beta, nu, X, Y, Time, MuX[io], VarX[io], MuY[io], VarY[io], g=it)
            rollback_dace(a[io], b[io], c[io], Time, U[io], V[io],
                          Dx, Dxx, MuX[io], VarX[io], Dy, Dyy, MuY[io], VarY[io], ResultE[io], gidx=it)
    result[:] = ResultE[:, indX, indY]


if __name__ == "__main__":

    sdfg = LocVolCalib_dace.to_sdfg()
    sdfg.simplify()
    auto_optimize(sdfg, dace.DeviceType.CPU)
    func = sdfg.compile()

    dataset_size = "L"
    outer, numX, numY, numT, s0, t, alpha, nu, beta = utils.getPrefedinedInputDataSet(dataset_size)

    #global array allocation
    numZ = max(numX, numY)
    a = np.ndarray((outer, numZ, numZ)).astype(np.float64)
    b = np.ndarray((outer, numZ, numZ)).astype(np.float64)
    c = np.ndarray((outer, numZ, numZ)).astype(np.float64)
    V = np.ndarray((outer, numX, numY)).astype(np.float64)
    U = np.ndarray((outer, numY, numX)).astype(np.float64)

    X = np.ndarray(numX).astype(np.float64)
    Dx = np.ndarray((numX, 3)).astype(np.float64)
    Dxx = np.ndarray((numX, 3)).astype(np.float64)
    Y = np.ndarray(numY).astype(np.float64)
    Dy = np.ndarray((numY, 3)).astype(np.float64)
    Dyy = np.ndarray((numY, 3)).astype(np.float64)
    Time = np.ndarray(numT).astype(np.float64)

    MuX = np.ndarray((outer, numY, numX)).astype(np.float64)
    MuY = np.ndarray((outer, numX, numY)).astype(np.float64)
    VarX = np.ndarray((outer, numY, numX)).astype(np.float64)
    VarY = np.ndarray((outer, numX, numY)).astype(np.float64)
    ResultE = np.ndarray((outer, numX, numY)).astype(np.float64)

    result = np.ndarray(outer).astype(np.float64)

    func(s0=s0, alpha=alpha, beta=beta, nu=nu, t=t,
         a=a, b=b, c=c, X=X, Y=Y, Time=Time,
         U=U, V=V, Dx=Dx, Dxx=Dxx, MuX=MuX, VarX=VarX, Dy=Dy, Dyy=Dyy, MuY=MuY, VarY=VarY,
         ResultE=ResultE, result=result, outer=outer, numXY=numZ, numX=numX, numY=numY, numT=numT)
    
    print("Computed results: ", result)
    print("------------")
    print("Expected results: ", utils.getPrefefinedOutputDataSet(dataset_size))
    print("------------")
    print(
        print(
            np.linalg.norm(result - utils.getPrefefinedOutputDataSet(dataset_size)) /
            np.linalg.norm(utils.getPrefefinedOutputDataSet(dataset_size))))
    # assert np.allclose(result, utils.getPrefefinedOutputDataSet(dataset_size))

    for i in range(10):
        start = time.perf_counter()
        func(s0=s0, alpha=alpha, beta=beta, nu=nu, t=t,
             a=a, b=b, c=c, X=X, Y=Y, Time=Time,
             U=U, V=V, Dx=Dx, Dxx=Dxx, MuX=MuX, VarX=VarX, Dy=Dy, Dyy=Dyy, MuY=MuY, VarY=VarY,
             ResultE=ResultE, result=result, outer=outer, numXY=numZ, numX=numX, numY=numY, numT=numT)
        end = time.perf_counter()
        print(f"Time in msecs: {(end-start)*1e3}")
