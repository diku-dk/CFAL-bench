import cupy as cp
import dace
import numpy as np
import time
import utils


from dace.transformation.auto.auto_optimize import apply_gpu_storage, auto_optimize


outer, numXY, numX, numY, numT = dace.symbol('outer'), dace.symbol('numXY'), dace.symbol('numX'), dace.symbol('numY'), dace.symbol('numT')
M, N = (dace.symbol(s) for s in ('M', 'N'))
g, gidx = dace.symbol('g'), dace.symbol('gidx')
xidx, yidx = dace.symbol('xidx'), dace.symbol('yidx')
# indX, indY = dace.symbol('indX'), dace.symbol('indY')


@dace.program
def initGrid_dace(s0: float, alpha: float, nu: float, t: float, indX: int, indY: int,
                  X: dace.float64[numX],
                  Y: dace.float64[numY],
                  Time: dace.float64[numT]):
    """ Initializes indX, indY, X, Y, and Time. """

    # Scalar computations
    # indX
    stdX = 20.0 * alpha * s0 * np.sqrt(t)
    dx = stdX / numX
    # indX = int(s0 / dx)
    # indY
    stdY = 10.0 * nu * np.sqrt(t)
    dy = stdY / numY
    logAlpha = np.log(alpha)
    # indY = int(numY / 2)

    # Array computations
    # Time
    for i in dace.map[0:numT]:
        Time[i] = t * i / (numT - 1)
    # X
    for i in dace.map[0:numX]:
        X[i] = i * np.log(i+1) * dx - indX * dx + s0
    # Y
    for i in dace.map[0:numY]:
        Y[i] = i * np.log(i+1) * dy - indY * dy + logAlpha

    # return indX, indY


@dace.program
def initOperator_dace(xx: dace.float64[N], D: dace.float64[3, N], DD: dace.float64[3, N]):
    """ Initializes Globals Dx[Dy] and Dxx[Dyy]. """

    dxul = xx[1] - xx[0]

    # lower boundary
    D[0, 0] = 0.0
    D[1, 0] = -1.0 / dxul
    D[2, 0] = 1.0 / dxul

    DD[0, 0] = 0.0
    DD[1, 0] = 0.0
    DD[2, 0] = 0.0

    #inner
    for i in dace.map[1:N - 1]:
        dxl = xx[i] - xx[i - 1]
        dxu = xx[i + 1] - xx[i]

        D[0, i] = -dxu / dxl / (dxl + dxu)
        D[1, i] = (dxu / dxl - dxl / dxu) / (dxl + dxu)
        D[2, i] = dxl / dxu / (dxl + dxu)

        DD[0, i] = 2.0 / dxl / (dxl + dxu)
        DD[1, i] = -2.0 * (1.0 / dxl + 1.0 / dxu) / (dxl + dxu)
        DD[2, i] = 2.0 / dxu / (dxl + dxu)

    # upper boundary
    dxlu = xx[N - 1] - xx[N - 2]

    D[0, N - 1] = -1.0 / dxlu
    D[1, N - 1] = 1.0 / dxlu
    D[2, N - 1] = 0.0

    DD[0, N - 1] = 0.0
    DD[1, N - 1] = 0.0
    DD[2, N - 1] = 0.0


num_outer, num_x, num_y, num_t = (dace.symbol(s) for s in ('num_outer', 'num_x', 'num_y', 'num_t'))


@dace.program
def initParams_dace(alpha: float, beta: float, nu: float,
                    X: dace.float64[num_x], Y: dace.float64[num_y], Time: dace.float64[num_t],
                    Dx: dace.float64[3, num_x], Dy: dace.float64[3, num_y],
                    Dxx: dace.float64[3, num_x], Dyy: dace.float64[3, num_y],
                    VarX: dace.float64[num_t, 3, num_x, num_y], VarY: dace.float64[num_t, 3, num_x, num_y],
                    b1: dace.float64[num_t, num_x, num_y], c1: dace.float64[num_t, num_x, num_y], beta1: dace.float64[num_t, num_x, num_y],
                    b2: dace.float64[num_t, num_y, num_x], c2: dace.float64[num_t, num_y, num_x], beta2: dace.float64[num_t, num_y, num_x]):

    dtInv = np.empty(num_t, dtype=dace.float64)
    dtInv[:-1] = 1.0 / (Time[1:] - Time[:-1])
    # for it in range(num_t - 2, -1, -1):
    #     var_x_tmp = np.exp(2 * np.add.outer(beta * np.log(X), Y - 0.5 * nu * nu * Time[it]))
    #     VarX[it, :, :, :] = var_x_tmp[np.newaxis, :, :] * Dxx[:, :, np.newaxis]
    #     VarY[it, :, :, :] = nu * nu * Dyy[:, np.newaxis, :]

    for it, i3, ix1, iy1 in dace.map[0:num_t, 0:3, 0:num_x, 0:num_y]:
        mu_x_tmp = 1e-7 / ((num_x + ix1) * (num_y + iy1))
        var_x_tmp = np.exp(2 * (beta * np.log(X[ix1]) + Y[iy1] - 0.5 * nu * nu * Time[it]))
        mu_y_tmp = alpha / (ix1 * num_y + iy1 + 1)
        var_y_tmp = (nu * nu) / (ix1 * num_y + iy1 + 1)
        VarX[it, i3, ix1, iy1] = mu_x_tmp * Dx[i3, ix1] + 0.5 * var_x_tmp * Dxx[i3, ix1]
        VarY[it, i3, ix1, iy1] = mu_y_tmp * Dy[i3, iy1] + 0.5 * var_y_tmp * Dyy[i3, iy1]

    # for ix in range(num_x):
    #     b1[:-1, ix, :] = dtInv[:-1, np.newaxis] - 0.25 * VarX[:-1, 1, ix, :]
    #     c1[:-1, ix, :] = -0.25 * VarX[:-1, 2, ix, :]
    #     if ix > 0:
    #         new_beta1 = -0.25 * VarX[:-1, 0, ix, :] / b1[:-1, ix - 1, :]
    #         b1[:-1, ix, :] = b1[:-1, ix, :] - new_beta1 * c1[:-1, ix - 1, :]
    #         beta1[:-1, ix, :] = new_beta1
    b1[:-1, 0, :] = dtInv[:-1, np.newaxis] - 0.5 * VarX[:-1, 1, 0, :]
    c1[:-1, 0, :] = -0.5 * VarX[:-1, 2, 0, :]
    for ix in range(1, num_x):
        b1[:-1, ix, :] = dtInv[:-1, np.newaxis] - 0.5 * VarX[:-1, 1, ix, :]
        c1[:-1, ix, :] = -0.5 * VarX[:-1, 2, ix, :]
        new_beta1 = -0.5 * VarX[:-1, 0, ix, :] / b1[:-1, ix - 1, :]
        b1[:-1, ix, :] = b1[:-1, ix, :] - new_beta1 * c1[:-1, ix - 1, :]
        beta1[:-1, ix, :] = new_beta1
    # for iy in range(num_y):
    #     b2[:-1, iy, :] = dtInv[:-1, np.newaxis] - 0.25 * VarY[:-1, 1, :, iy]
    #     c2[:-1, iy, :] = -0.25 * VarY[:-1, 2, :, iy]
    #     if iy > 0:
    #         new_beta2 = -0.25 * VarY[:-1, 0, :, iy] / b2[:-1, iy - 1, :]
    #         b2[:-1, iy, :] = b2[:-1, iy, :] - new_beta2 * c2[:-1, iy - 1, :]
    #         beta2[:-1, iy, :] = new_beta2
    b2[:-1, 0, :] = dtInv[:-1, np.newaxis] - 0.5 * VarY[:-1, 1, :, 0]
    c2[:-1, 0, :] = -0.5 * VarY[:-1, 2, :, 0]
    for iy in range(1, num_y):
        b2[:-1, iy, :] = dtInv[:-1, np.newaxis] - 0.5 * VarY[:-1, 1, :, iy]
        c2[:-1, iy, :] = -0.5 * VarY[:-1, 2, :, iy]
        new_beta2 = -0.5 * VarY[:-1, 0, :, iy] / b2[:-1, iy - 1, :]
        b2[:-1, iy, :] = b2[:-1, iy, :] - new_beta2 * c2[:-1, iy - 1, :]
        beta2[:-1, iy, :] = new_beta2

# @dace.program
# def initParams_dace(beta: float, nu: float,
#                     X: dace.float64[num_x], Y: dace.float64[num_y], Time: dace.float64[num_t],
#                     Dxx: dace.float64[3, num_x], Dyy: dace.float64[3, num_y],
#                     VarX: dace.float64[num_t, 3, num_x, num_y], VarY: dace.float64[num_t, 3, num_x, num_y],
#                     b1: dace.float64[num_t, num_x, num_y], c1: dace.float64[num_t, num_x, num_y], beta1: dace.float64[num_t, num_x, num_y],
#                     b2: dace.float64[num_t, num_y, num_x], c2: dace.float64[num_t, num_y, num_x], beta2: dace.float64[num_t, num_y, num_x]):

#     dtInv = np.empty(num_t, dtype=dace.float64)
#     dtInv[:-1] = 1.0 / (Time[1:] - Time[:-1])
#     # for it in range(num_t - 2, -1, -1):
#     #     var_x_tmp = np.exp(2 * np.add.outer(beta * np.log(X), Y - 0.5 * nu * nu * Time[it]))
#     #     VarX[it, :, :, :] = var_x_tmp[np.newaxis, :, :] * Dxx[:, :, np.newaxis]
#     #     VarY[it, :, :, :] = nu * nu * Dyy[:, np.newaxis, :]

#     for it, i3, ix1, iy1 in dace.map[0:num_t, 0:3, 0:num_x, 0:num_y]:
#         var_x_tmp = np.exp(2 * (beta * np.log(X[ix1]) + Y[iy1] - 0.5 * nu * nu * Time[it]))
#         VarX[it, i3, ix1, iy1] = var_x_tmp * Dxx[i3, ix1]
#         VarY[it, i3, ix1, iy1] = nu * nu * Dyy[i3, iy1]

#     # for ix in range(num_x):
#     #     b1[:-1, ix, :] = dtInv[:-1, np.newaxis] - 0.25 * VarX[:-1, 1, ix, :]
#     #     c1[:-1, ix, :] = -0.25 * VarX[:-1, 2, ix, :]
#     #     if ix > 0:
#     #         new_beta1 = -0.25 * VarX[:-1, 0, ix, :] / b1[:-1, ix - 1, :]
#     #         b1[:-1, ix, :] = b1[:-1, ix, :] - new_beta1 * c1[:-1, ix - 1, :]
#     #         beta1[:-1, ix, :] = new_beta1
#     b1[:-1, 0, :] = dtInv[:-1, np.newaxis] - 0.25 * VarX[:-1, 1, 0, :]
#     c1[:-1, 0, :] = -0.25 * VarX[:-1, 2, 0, :]
#     for ix in range(1, num_x):
#         b1[:-1, ix, :] = dtInv[:-1, np.newaxis] - 0.25 * VarX[:-1, 1, ix, :]
#         c1[:-1, ix, :] = -0.25 * VarX[:-1, 2, ix, :]
#         new_beta1 = -0.25 * VarX[:-1, 0, ix, :] / b1[:-1, ix - 1, :]
#         b1[:-1, ix, :] = b1[:-1, ix, :] - new_beta1 * c1[:-1, ix - 1, :]
#         beta1[:-1, ix, :] = new_beta1
#     # for iy in range(num_y):
#     #     b2[:-1, iy, :] = dtInv[:-1, np.newaxis] - 0.25 * VarY[:-1, 1, :, iy]
#     #     c2[:-1, iy, :] = -0.25 * VarY[:-1, 2, :, iy]
#     #     if iy > 0:
#     #         new_beta2 = -0.25 * VarY[:-1, 0, :, iy] / b2[:-1, iy - 1, :]
#     #         b2[:-1, iy, :] = b2[:-1, iy, :] - new_beta2 * c2[:-1, iy - 1, :]
#     #         beta2[:-1, iy, :] = new_beta2
#     b2[:-1, 0, :] = dtInv[:-1, np.newaxis] - 0.25 * VarY[:-1, 1, :, 0]
#     c2[:-1, 0, :] = -0.25 * VarY[:-1, 2, :, 0]
#     for iy in range(1, num_y):
#         b2[:-1, iy, :] = dtInv[:-1, np.newaxis] - 0.25 * VarY[:-1, 1, :, iy]
#         c2[:-1, iy, :] = -0.25 * VarY[:-1, 2, :, iy]
#         new_beta2 = -0.25 * VarY[:-1, 0, :, iy] / b2[:-1, iy - 1, :]
#         b2[:-1, iy, :] = b2[:-1, iy, :] - new_beta2 * c2[:-1, iy - 1, :]
#         beta2[:-1, iy, :] = new_beta2


@dace.program
def value_dace(Time: dace.float64[num_t],
               VarX: dace.float64[num_t, 3, num_x, num_y], VarY: dace.float64[num_t, 3, num_x, num_y],
               b1: dace.float64[num_t, num_x, num_y], c1: dace.float64[num_t, num_x, num_y], beta1: dace.float64[num_t, num_x, num_y],
               b2: dace.float64[num_t, num_y, num_x], c2: dace.float64[num_t, num_y, num_x], beta2: dace.float64[num_t, num_y, num_x],
               ResultE: dace.float64[num_x, num_y, num_outer]):

    U = np.empty((num_x, num_y, num_outer), dtype=np.float64)
    V = np.empty((num_x, num_y, num_outer), dtype=np.float64)

    for it in range(num_t - 2, -1, -1):
        dtInv = 1.0 / (Time[it + 1] - Time[it])

        U[:] = (dtInv + 0.5 * VarX[it, 1, :, :, np.newaxis]) * ResultE[:, :, :]
        U[1:-1] += (0.5 * VarX[it, 0, 1:-1, :, np.newaxis] * ResultE[:-2, :, :] +
                    0.5 * VarX[it, 2, 1:-1, :, np.newaxis] * ResultE[2:, :, :])

        V[:] = 0.5 * VarY[it, 1, :, :, np.newaxis] * ResultE[:, :, :]
        V[:, 1:-1] += (0.5 * VarY[it, 0, :, 1:-1, np.newaxis] * ResultE[:, :-2, :] +
                       0.5 * VarY[it, 2, :, 1:-1, np.newaxis] * ResultE[:, 2:, :])

        U[:] += V
        
        for ix in range(1, num_x - 1):
            U[ix, :, :] -= beta1[it, ix, :, np.newaxis] * U[ix - 1, :, :]
        U[num_x - 1, :, :] = U[num_x - 1, :, :] / b1[it, num_x - 1, :, np.newaxis]
        for ix in range(num_x - 2, -1, -1):
            U[ix, :, :] = (U[ix, :, :] - c1[it, ix, :, np.newaxis] * U[ix + 1, :, :]) / b1[it, ix, :, np.newaxis]

        ResultE[:, 0, :] = dtInv * U[:, 0, :] - 0.5 * V[:, 0, :]
        for iy in range(1, num_y - 1):
            ResultE[:, iy, :] = dtInv * U[:, iy, :] - 0.5 * V[:, iy, :] - beta2[it, iy, :, np.newaxis] * ResultE[:, iy - 1, :]
        ResultE[:, num_y - 1, :] = (dtInv * U[:, num_y - 1, :] - 0.5 * V[:, num_y - 1, :]) / b2[it, num_y - 1, :, np.newaxis]
        for iy in range(num_y - 2, -1, -1):
            ResultE[:, iy, :] = (ResultE[:, iy, :] - c2[it, iy, :, np.newaxis] * ResultE[:, iy + 1, :]) / b2[it, iy, :, np.newaxis]
    

# @dace.program
# def value_dace(Time: dace.float64[num_t],
#                VarX: dace.float64[num_t, 3, num_x, num_y], VarY: dace.float64[num_t, 3, num_x, num_y],
#                b1: dace.float64[num_t, num_x, num_y], c1: dace.float64[num_t, num_x, num_y], beta1: dace.float64[num_t, num_x, num_y],
#                b2: dace.float64[num_t, num_y, num_x], c2: dace.float64[num_t, num_y, num_x], beta2: dace.float64[num_t, num_y, num_x],
#                ResultE: dace.float64[num_x, num_y, num_outer]):

#     U = np.empty((num_x, num_y, num_outer), dtype=np.float64)
#     V = np.empty((num_x, num_y, num_outer), dtype=np.float64)

#     for it in range(num_t - 2, -1, -1):
#         dtInv = 1.0 / (Time[it + 1] - Time[it])

#         U[:] = (dtInv + 0.25 * VarX[it, 1, :, :, np.newaxis]) * ResultE[:, :, :]
#         U[1:-1] += (0.25 * VarX[it, 0, 1:-1, :, np.newaxis] * ResultE[:-2, :, :] +
#                     0.25 * VarX[it, 2, 1:-1, :, np.newaxis] * ResultE[2:, :, :])

#         V[:] = 0.5 * VarY[it, 1, :, :, np.newaxis] * ResultE[:, :, :]
#         V[:, 1:-1] += (0.5 * VarY[it, 0, :, 1:-1, np.newaxis] * ResultE[:, :-2, :] +
#                        0.5 * VarY[it, 2, :, 1:-1, np.newaxis] * ResultE[:, 2:, :])

#         U[:] += V
        
#         for ix in range(1, num_x - 1):
#             U[ix, :, :] -= beta1[it, ix, :, np.newaxis] * U[ix - 1, :, :]
#         U[num_x - 1, :, :] = U[num_x - 1, :, :] / b1[it, num_x - 1, :, np.newaxis]
#         for ix in range(num_x - 2, -1, -1):
#             U[ix, :, :] = (U[ix, :, :] - c1[it, ix, :, np.newaxis] * U[ix + 1, :, :]) / b1[it, ix, :, np.newaxis]

#         ResultE[:, 0, :] = dtInv * U[:, 0, :] - 0.5 * V[:, 0, :]
#         for iy in range(1, num_y - 1):
#             ResultE[:, iy, :] = dtInv * U[:, iy, :] - 0.5 * V[:, iy, :] - beta2[it, iy, :, np.newaxis] * ResultE[:, iy - 1, :]
#         ResultE[:, num_y - 1, :] = (dtInv * U[:, num_y - 1, :] - 0.5 * V[:, num_y - 1, :]) / b2[it, num_y - 1, :, np.newaxis]
#         for iy in range(num_y - 2, -1, -1):
#             ResultE[:, iy, :] = (ResultE[:, iy, :] - c2[it, iy, :, np.newaxis] * ResultE[:, iy + 1, :]) / b2[it, iy, :, np.newaxis]
        

@dace.program
def setPayoff_dace(strike: float, X: dace.float64[numX], ResultE: dace.float64[numX, numY, 1]):

    for ip in dace.map[0:numX]:
        ResultE[ip, :] = max(X[ip] - strike, 0.0)
    # ResultE[:] = np.reshape(np.maximum(X - strike, 0.0), (numX, 1))


@dace.program
def LocVolCalib_dace(s0: float, alpha: float, beta: float, nu: float, t: float, indX: int, indY: int, result: dace.float64[num_outer]):
    
    X = np.empty(num_x, dtype=np.float64)
    Y = np.empty(num_y, dtype=np.float64)
    Time = np.empty(num_t, dtype=np.float64)
    Dx = np.empty((3, num_x), dtype=np.float64)
    Dxx = np.empty((3, num_x), dtype=np.float64)
    Dy = np.empty((3, num_y), dtype=np.float64)
    Dyy = np.empty((3, num_y), dtype=np.float64)
    VarX = np.empty((num_t, 3, num_x, num_y), dtype=np.float64)
    VarY = np.empty((num_t, 3, num_x, num_y), dtype=np.float64)
    b1 = np.empty((num_t, num_x, num_y), dtype=np.float64)
    c1 = np.empty((num_t, num_x, num_y), dtype=np.float64)
    beta1 = np.empty((num_t, num_x, num_y), dtype=np.float64)
    b2 = np.empty((num_t, num_y, num_x), dtype=np.float64)
    c2 = np.empty((num_t, num_y, num_x), dtype=np.float64)
    beta2 = np.empty((num_t, num_y, num_x), dtype=np.float64)
    ResultE = np.empty((num_x, num_y, num_outer), dtype=np.float64)
    
    initGrid_dace(s0, alpha, nu, t, indX, indY, X, Y, Time)
    initOperator_dace(X, Dx, Dxx)
    initOperator_dace(Y, Dy, Dyy)
    # initParams_dace(beta, nu, X, Y, Time, Dxx, Dyy, VarX, VarY, b1, c1, beta1, b2, c2, beta2)
    initParams_dace(alpha, beta, nu, X, Y, Time, Dx, Dy, Dxx, Dyy, VarX, VarY, b1, c1, beta1, b2, c2, beta2)
    for ix0, io in dace.map[0:num_x, 0:num_outer]:
        ResultE[ix0, :, io] = max(X[ix0] - 0.001 * np.float64(io), 0.0)
    value_dace(Time, VarX, VarY, b1, c1, beta1, b2, c2, beta2, ResultE)
    result[:] = ResultE[indX, indY, :]


if __name__ == "__main__":

    sdfg = LocVolCalib_dace.to_sdfg(simplify=False)
    sdfg.simplify()
    auto_optimize(sdfg, dace.DeviceType.CPU)
    # auto_optimize(sdfg, dace.DeviceType.GPU, use_gpu_storage=True)
    func = sdfg.compile()

    dataset_size = "L"
    outer, numX, numY, numT, s0, t, alpha, nu, beta = utils.getPrefedinedInputDataSet(dataset_size)

    X = np.ndarray(numX).astype(np.float64)
    Y = np.ndarray(numY).astype(np.float64)
    Time = np.ndarray(numT).astype(np.float64)


    result = np.empty(outer, dtype=np.float64)

    # idx, idy = initGrid_dace(s0, alpha, nu, t, X, Y, Time)
    # print(idx, idy)

    # indX
    stdX = 20.0 * alpha * s0 * np.sqrt(t)
    dx = stdX / numX
    indX = int(s0 / dx)
    # indY
    stdY = 10.0 * nu * np.sqrt(t)
    dy = stdY / numY
    logAlpha = np.log(alpha)
    indY = int(numY / 2)


    func(s0=s0, alpha=alpha, beta=beta, nu=nu, t=t, indX=indX, indY=indY, result=result,
         num_outer=outer, num_x=numX, num_y=numY, num_t=numT)
    
    print("Computed results: ", result)
    print("------------")
    print("Expected results: ", utils.getPrefefinedOutputDataSet(dataset_size))
    print("------------")
    # print(
    #     print(
    #         np.linalg.norm(result - utils.getPrefefinedOutputDataSet(dataset_size)) /
    #         np.linalg.norm(utils.getPrefefinedOutputDataSet(dataset_size))))
    # assert np.allclose(result, utils.getPrefefinedOutputDataSet(dataset_size))

    for i in range(10):
        start = time.perf_counter()
        func(s0=s0, alpha=alpha, beta=beta, nu=nu, t=t, indX=indX, indY=indY, result=result,
             num_outer=outer, num_x=numX, num_y=numY, num_t=numT)
        end = time.perf_counter()
        print(f"Time in msecs: {(end-start)*1e3}")
