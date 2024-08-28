import argparse
import copy
import cupy as cp
import dace
import numpy as np
import utils

from timeit import repeat
from dace.transformation.auto.auto_optimize import apply_gpu_storage, auto_optimize, greedy_fuse
from dace.transformation.dataflow import MapExpansion, MapFusion, MapCollapse


outer, numXY, numX, numY, numT = dace.symbol('outer'), dace.symbol('numXY'), dace.symbol('numX'), dace.symbol('numY'), dace.symbol('numT')
M, N = (dace.symbol(s) for s in ('M', 'N'))
g, gidx = dace.symbol('g'), dace.symbol('gidx')
xidx, yidx = dace.symbol('xidx'), dace.symbol('yidx')


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
                    MuVarX: dace.float64[num_t, 3, num_x, num_y], MuVarY: dace.float64[num_t, 3, num_x, num_y],
                    b1: dace.float64[num_t, num_x, num_y], c1: dace.float64[num_t, num_x, num_y], beta1: dace.float64[num_t, num_x, num_y],
                    b2: dace.float64[num_t, num_x, num_y], c2: dace.float64[num_t, num_x, num_y], beta2: dace.float64[num_t, num_x, num_y]):

    dtInv = np.empty(num_t, dtype=dace.float64)
    dtInv[:-1] = 1.0 / (Time[1:] - Time[:-1])
    
    for it, i3, ix1, iy1 in dace.map[0:num_t, 0:3, 0:num_x, 0:num_y]:
        mu_x_tmp = 1e-7 / ((num_x + ix1) * (num_y + iy1))
        var_x_tmp = np.exp(2 * (beta * np.log(X[ix1]) + Y[iy1] - 0.5 * nu * nu * Time[it]))
        mu_y_tmp = alpha / (ix1 * num_y + iy1 + 1)
        var_y_tmp = (nu * nu) / (ix1 * num_y + iy1 + 1)
        MuVarX[it, i3, ix1, iy1] = 0.5 * mu_x_tmp * Dx[i3, ix1] + 0.25 * var_x_tmp * Dxx[i3, ix1]
        MuVarY[it, i3, ix1, iy1] = 0.5 * mu_y_tmp * Dy[i3, iy1] + 0.25 * var_y_tmp * Dyy[i3, iy1]
    
    for it in dace.map[0:num_t]:
        MuVarX[it, 0, 0, :] = 0
        MuVarX[it, 0, -1, :] = 0
        MuVarX[it, 2, 0, :] = 0
        MuVarX[it, 2, -1, :] = 0
        MuVarY[it, 0, :, 0] = 0
        MuVarY[it, 0, :, -1] = 0
        MuVarY[it, 2, :, 0] = 0
        MuVarY[it, 2, :, -1] = 0

    b1[:-1, 0, :] = dtInv[:-1, np.newaxis] - MuVarX[:-1, 1, 0, :]
    c1[:-1, 0, :] = -MuVarX[:-1, 2, 0, :]
    for ix in range(1, num_x):
        b1[:-1, ix, :] = dtInv[:-1, np.newaxis] - MuVarX[:-1, 1, ix, :]
        c1[:-1, ix, :] = - MuVarX[:-1, 2, ix, :]
        new_beta1 = - MuVarX[:-1, 0, ix, :] / b1[:-1, ix - 1, :]
        b1[:-1, ix, :] = b1[:-1, ix, :] - new_beta1 * c1[:-1, ix - 1, :]
        beta1[:-1, ix, :] = new_beta1
    c1[:-1, -1, :] = 0

    b2[:-1, :, 0] = dtInv[:-1, np.newaxis] - MuVarY[:-1, 1, :, 0]
    c2[:-1, :, 0] = -MuVarY[:-1, 2, :, 0]
    for iy in range(1, num_y):
        b2[:-1, :, iy] = dtInv[:-1, np.newaxis] - MuVarY[:-1, 1, :, iy]
        c2[:-1, :, iy] = - MuVarY[:-1, 2, :, iy]
        new_beta2 = - MuVarY[:-1, 0, :, iy] / b2[:-1, :, iy - 1]
        b2[:-1, :, iy] = b2[:-1, :, iy] - new_beta2 * c2[:-1, :, iy - 1]
        beta2[:-1, :, iy] = new_beta2
    c2[:-1, :, -1] = 0
    beta2[:-1, :, 0] = 0


@dace.program
def value_dace(Time: dace.float64[num_t],
               MuVarX: dace.float64[num_t, 3, num_x, num_y], MuVarY: dace.float64[num_t, 3, num_x, num_y],
               b1: dace.float64[num_t, num_x, num_y], c1: dace.float64[num_t, num_x, num_y], beta1: dace.float64[num_t, num_x, num_y],
               b2: dace.float64[num_t, num_x, num_y], c2: dace.float64[num_t, num_x, num_y], beta2: dace.float64[num_t, num_x, num_y],
               ResE: dace.float64[num_x + 2, num_y + 2, num_outer]):

    U = np.empty((num_x, num_y, num_outer), dtype=np.float64)
    V = np.empty((num_x, num_y, num_outer), dtype=np.float64)
    ResultE = ResE[1:-1, 1:-1, :]

    for it in range(num_t - 2, -1, -1):
        dtInv = 1.0 / (Time[it + 1] - Time[it])

        U[:] = ((dtInv + MuVarX[it, 1, :, :, np.newaxis]) * ResultE[:, :, :] +
                MuVarX[it, 0, :, :, np.newaxis] * ResE[:-2, 1:-1, :] +
                MuVarX[it, 2, :, :, np.newaxis] * ResE[2:, 1:-1, :])

        V[:] = (MuVarY[it, 1, :, :, np.newaxis] * ResultE[:, :, :] +
                MuVarY[it, 0, :, :, np.newaxis] * ResE[1:-1, :-2, :] +
                MuVarY[it, 2, :, :, np.newaxis] * ResE[1:-1, 2:, :])

        U2 = U + V

        for ix in range(1, num_x - 1):
            U2[ix, :, :] -= beta1[it, ix, :, np.newaxis] * U2[ix - 1, :, :]
        U2[num_x - 1, :, :] = U2[num_x - 1, :, :] / b1[it, num_x - 1, :, np.newaxis]
        for ix in range(num_x - 2, -1, -1):
            for iy2, iouter in dace.map[0:num_y, 0:num_outer]:
                U2[ix, iy2, iouter] = (U2[ix, iy2, iouter] - c1[it, ix, iy2] * U2[ix + 1, iy2, iouter]) / b1[it, ix, iy2]

        for iy in range(num_y):
            ResultE[:, iy, :] = dtInv * U2[:, iy, :] - 0.5 * V[:, iy, :] - beta2[it, :, iy, np.newaxis] * ResultE[:, iy - 1, :]
        for iy in range(num_y - 1, -1, -1):
            for ix2, iouter1 in dace.map[0:num_x, 0:num_outer]:
                ResultE[ix2, iy, iouter1] = (ResultE[ix2, iy, iouter1] - c2[it, ix2, iy] * ResultE[ix2, iy + 1, iouter1]) / b2[it, ix2, iy]  


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
    MuVarX = np.empty((num_t, 3, num_x, num_y), dtype=np.float64)
    MuVarY = np.empty((num_t, 3, num_x, num_y), dtype=np.float64)
    b1 = np.empty((num_t, num_x, num_y), dtype=np.float64)
    c1 = np.empty((num_t, num_x, num_y), dtype=np.float64)
    beta1 = np.empty((num_t, num_x, num_y), dtype=np.float64)
    b2 = np.empty((num_t, num_x, num_y), dtype=np.float64)
    c2 = np.empty((num_t, num_x, num_y), dtype=np.float64)
    beta2 = np.empty((num_t, num_x, num_y), dtype=np.float64)
    ResE = np.empty((num_x + 2, num_y + 2, num_outer), dtype=np.float64)
    ResultE = ResE[1:-1, 1:-1, :]
    
    initGrid_dace(s0, alpha, nu, t, indX, indY, X, Y, Time)
    initOperator_dace(X, Dx, Dxx)
    initOperator_dace(Y, Dy, Dyy)
    initParams_dace(alpha, beta, nu, X, Y, Time, Dx, Dy, Dxx, Dyy, MuVarX, MuVarY, b1, c1, beta1, b2, c2, beta2)
    for ix0, io in dace.map[0:num_x, 0:num_outer]:
        ResultE[ix0, :, io] = max(X[ix0] - 0.001 * np.float64(io), 0.0)
    value_dace(Time, MuVarX, MuVarY, b1, c1, beta1, b2, c2, beta2, ResE)
    result[:] = ResultE[indX, indY, :]


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--target", type=str, choices=["cpu", "gpu"], default="cpu")
    argparser.add_argument("--size", type=str, choices=["S", "M", "L"], default="S")
    args = vars(argparser.parse_args())

    print(f"Running LocVolCalib with {'GPU' if args['target'] == 'gpu' else 'CPU'} target and dataset size {args['size']} ...", flush=True)

    dataset_size = args["size"]
    outer, numX, numY, numT, s0, t, alpha, nu, beta = utils.getPrefedinedInputDataSet(dataset_size)
    ref = utils.getPrefefinedOutputDataSet(dataset_size)

    sdfg = LocVolCalib_dace.to_sdfg(simplify=False)
    sdfg.simplify()

    if args["target"] == "gpu":
        auto_optimize(sdfg, dace.DeviceType.GPU, use_gpu_storage=True)
        result = cp.empty(outer, dtype=np.float64)
    else:
        # First version
        sdfg_1 = copy.deepcopy(sdfg)
        auto_optimize(sdfg_1, dace.DeviceType.CPU, move_loop_into_map=True)

        # Second version (make batch parallelism the outermost loop)
        # Change strides of data
        for name, arr in sdfg.arrays.items():
            found = False
            idx = -1
            for i, s in enumerate(arr.shape):
                if dace.symbolic.issymbolic(s) and num_outer in s.free_symbols:
                    found = True
                    idx = i
                    break
            if found and idx > 0:
                extent = arr.shape[idx]
                new_strides = [s / extent if i != idx else s for i, s in enumerate(arr.strides)]
                new_strides[idx] = new_strides[0] * arr.shape[0]
                arr.strides = new_strides
                print(f"Array {name} shape: {arr.shape}, strides: {arr.strides}")
        # Auto-opt
        auto_optimize(sdfg, dace.DeviceType.CPU, move_loop_into_map=True)
        # Find maps with batch parallelism and permute them
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry) and len(node.map.params) > 1:
                found = False
                idx = -1
                for i, r in enumerate(node.map.range):
                    syms = (str(s) for s in r[1].free_symbols)
                    if 'num_outer' in syms:
                        found = True
                        idx = i
                        break
                if found:
                    node.map.params = [node.map.params[idx]] + node.map.params[:idx] + node.map.params[idx + 1:]
                    node.map.range = [node.map.range[idx]] + node.map.range[:idx] + node.map.range[idx + 1:]
                    MapExpansion.apply_to(sdfg, map_entry=node)
        # Fuse again
        greedy_fuse(sdfg, validate_all=True)
        sdfg.apply_transformations_repeated([MapCollapse])

        result = np.empty(outer, dtype=np.float64)

    func = sdfg.compile()

    # Compute indX/indY to work around bug in GPU codegen
    X = np.ndarray(numX).astype(np.float64)
    Y = np.ndarray(numY).astype(np.float64)
    Time = np.ndarray(numT).astype(np.float64)
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
    if args["target"] == "gpu":
        val = cp.asnumpy(result)
    else:
        val = result
    
    print(f"Relative error: {np.linalg.norm(val - ref) / np.linalg.norm(ref)}")
    
    def _func():
        func(s0=s0, alpha=alpha, beta=beta, nu=nu, t=t, indX=indX, indY=indY, result=result,
             num_outer=outer, num_x=numX, num_y=numY, num_t=numT)

    num_warmup = 2

    for _ in range(num_warmup):
        _func()
    runtimes = repeat(lambda: _func(), number=1, repeat=10)
    mean = np.mean(runtimes)
    std = np.std(runtimes)
    repeatitions = 10
    while std > 0.01 * mean and len(runtimes) < 100:
        print(f"Standard deviation too high ({std * 100 / mean:.2f}% of the mean) after {repeatitions} repeatitions ...", flush=True)
        runtimes.extend(repeat(lambda: _func(), number=1, repeat=10))
        mean = np.mean(runtimes)
        std = np.std(runtimes)
        repeatitions += 10
    print(f"DaCe {args['target'].upper()} runtime: mean {mean} s, std {std * 100 / mean:.2f}%", flush=True)

    if args["target"] == "cpu":
        func = sdfg_1.compile()
        result = np.empty(outer, dtype=np.float64)
        func(s0=s0, alpha=alpha, beta=beta, nu=nu, t=t, indX=indX, indY=indY, result=result,
             num_outer=outer, num_x=numX, num_y=numY, num_t=numT)
        val = result
        print(f"Relative error: {np.linalg.norm(val - ref) / np.linalg.norm(ref)}")
        
        def _func():
            func(s0=s0, alpha=alpha, beta=beta, nu=nu, t=t, indX=indX, indY=indY, result=result,
                num_outer=outer, num_x=numX, num_y=numY, num_t=numT)

        num_warmup = 2

        for _ in range(num_warmup):
            _func()
        runtimes = repeat(lambda: _func(), number=1, repeat=10)
        mean = np.mean(runtimes)
        std = np.std(runtimes)
        repeatitions = 10
        while std > 0.01 * mean and len(runtimes) < 100:
            print(f"Standard deviation too high ({std * 100 / mean:.2f}% of the mean) after {repeatitions} repeatitions ...", flush=True)
            runtimes.extend(repeat(lambda: _func(), number=1, repeat=10))
            mean = np.mean(runtimes)
            std = np.std(runtimes)
            repeatitions += 10
        print(f"DaCe CPU version 2 runtime: mean {mean} s, std {std * 100 / mean:.2f}%", flush=True)
