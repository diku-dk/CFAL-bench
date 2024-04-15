import copy
import dace
import numpy as np
import numpy.typing as npt


from test_tridiag import tridiag_cpython, tridiag_dace
from dace.transformation.auto.auto_optimize import apply_gpu_storage, auto_optimize


def rollback_cpython(numX: int, numY: int, g: int,
                     a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], c: npt.NDArray[np.float64],
                     Time: npt.NDArray[np.float64], U: npt.NDArray[np.float64], V: npt.NDArray[np.float64],
                     Dx: npt.NDArray[np.float64], Dxx: npt.NDArray[np.float64],
                     MuX: npt.NDArray[np.float64], VarX: npt.NDArray[np.float64],
                     Dy: npt.NDArray[np.float64], Dyy: npt.NDArray[np.float64],
                     MuY: npt.NDArray[np.float64], VarY: npt.NDArray[np.float64],
                     ResultE: npt.NDArray[np.float64]):

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
            a[j, i] = -0.5 * (MuX[j, i] * Dx[i, 0] + 0.5 * VarX[j, i] * Dxx[i, 0])
            b[j, i] = dtInv - 0.5 * (MuX[j, i] * Dx[i, 1] + 0.5 * VarX[j, i] * Dxx[i, 1])
            c[j, i] = -0.5 * (MuX[j, i] * Dx[i, 2] + 0.5 * VarX[j, i] * Dxx[i, 2])

        uu = U[j, :]

        tridiag_cpython(numX, a[j], b[j], c[j], uu)

    # implicit y
    for i in range(numX):
        for j in range(numY):
            a[i, j] = -0.5 * (MuY[i, j] * Dy[j, 0] + 0.5 * VarY[i, j] * Dyy[j, 0])
            b[i, j] = dtInv - 0.5 * (MuY[i, j] * Dy[j, 1] + 0.5 * VarY[i, j] * Dyy[j, 1])
            c[i, j] = -0.5 * (MuY[i, j] * Dy[j, 2] + 0.5 * VarY[i, j] * Dyy[j, 2])

        yy = ResultE[i, :]

        for j in range(numY):
            yy[j] = dtInv * U[j, i] - 0.5 * V[i, j]

        tridiag_cpython(numY, a[i], b[i], c[i], yy)


numXY, numX, numY, numT = dace.symbol('numXY'), dace.symbol('numX'), dace.symbol('numY'), dace.symbol('numT')
gidx = dace.symbol('gidx')


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
        U[j, i] = dtInv * ResultE[i, j] + 0.5 * ResultE[i, j] * (MuX[j, i] * Dx[i, 1] + 0.5 * VarX[j, i] * Dxx[i, 1])
        if i > 0:
            U[j, i] = U[j, i] + 0.5 * ResultE[i - 1, j] * (MuX[j, i] * Dx[i, 0] + 0.5 * VarX[j, i] * Dxx[i, 0])
        if i < numX - 1:
            U[j, i] = U[j, i] + 0.5 * ResultE[i + 1, j] * (MuX[j, i] * Dx[i, 2] + 0.5 * VarX[j, i] * Dxx[i, 2])

    # explicit y
    for i, j in dace.map[0:numX, 0:numY]:
        V[i, j] = ResultE[i, j] * (MuY[i, j] * Dy[j, 1] + 0.5 * VarY[i, j] * Dyy[j, 1])
        if j > 0:
            V[i, j] = V[i, j] + ResultE[i, j - 1] * (MuY[i, j] * Dy[j, 0] + 0.5 * VarY[i, j] * Dyy[j, 0])
        if j < numY - 1:
            V[i, j] = V[i, j] + ResultE[i, j + 1] * (MuY[i, j] * Dy[j, 2] + 0.5 * VarY[i, j] * Dyy[j, 2])
        U[j, i] = U[j, i] + V[i, j]

    # implicit x
    for j in dace.map[0:numY]:
        for i in range(numX):
            a[j, i] = -0.5 * (MuX[j, i] * Dx[i, 0] + 0.5 * VarX[j, i] * Dxx[i, 0])
            b[j, i] = dtInv - 0.5 * (MuX[j, i] * Dx[i, 1] + 0.5 * VarX[j, i] * Dxx[i, 1])
            c[j, i] = -0.5 * (MuX[j, i] * Dx[i, 2] + 0.5 * VarX[j, i] * Dxx[i, 2])

        # uu = U[j, :]

        tridiag_dace(a[j, :numX], b[j, :numX], c[j, :numX], U[j, :])

    # implicit y
    for i in dace.map[0:numX]:
        for j in range(numY):
            a[i, j] = -0.5 * (MuY[i, j] * Dy[j, 0] + 0.5 * VarY[i, j] * Dyy[j, 0])
            b[i, j] = dtInv - 0.5 * (MuY[i, j] * Dy[j, 1] + 0.5 * VarY[i, j] * Dyy[j, 1])
            c[i, j] = -0.5 * (MuY[i, j] * Dy[j, 2] + 0.5 * VarY[i, j] * Dyy[j, 2])
            ResultE[i, j] = dtInv * U[j, i] - 0.5 * V[i, j]

        # yy = ResultE[i, :]

        # for j in range(numY):
        #     yy[j] = dtInv * U[j, i] - 0.5 * V[i, j]

        tridiag_dace(a[i, :numY], b[i, :numY], c[i, :numY], ResultE[i, :])


def validate(U_ref: npt.NDArray[np.float64], V_ref: npt.NDArray[np.float64], ResultE_ref,
             U_val: npt.NDArray[np.float64], V_val: npt.NDArray[np.float64], ResultE_val: npt.NDArray[np.float64]):
        assert np.allclose(U_ref, U_val)
        assert np.allclose(V_ref, V_val)
        assert np.allclose(ResultE_ref, ResultE_val)


if __name__ == "__main__":

    seed = 42
    rng = np.random.default_rng(seed)

    numX = 1000
    numY = 1000
    numT = 1000
    numXY = max(numX, numY)

    g = rng.integers(0, numT-1)
    Time = rng.random(numT)
    Dx = rng.random((numX, 3))
    Dxx = rng.random((numX, 3))
    Mux = rng.random((numY, numX))
    Varx = rng.random((numY, numX))
    Dy = rng.random((numY, 3))
    Dyy = rng.random((numY, 3))
    Muy = rng.random((numX, numY))
    Vary = rng.random((numX, numY))
    ResultE = rng.random((numX, numY))

    a_ref = np.empty((numXY, numXY), dtype=np.float64)
    b_ref  = np.empty((numXY, numXY), dtype=np.float64)
    c_ref = np.empty((numXY, numXY), dtype=np.float64)
    U_ref = np.empty((numY, numX), dtype=np.float64)
    V_ref = np.empty((numX, numY), dtype=np.float64)
    ResultE_ref = copy.deepcopy(ResultE)
    rollback_cpython(numX, numY, g, a_ref, b_ref, c_ref, Time, U_ref, V_ref, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_ref)

    a_val = np.empty((numXY, numXY), dtype=np.float64)
    b_val = np.empty((numXY, numXY), dtype=np.float64)
    c_val = np.empty((numXY, numXY), dtype=np.float64)
    U_val = np.empty((numY, numX), dtype=np.float64)
    V_val = np.empty((numX, numY), dtype=np.float64)

    naive = rollback_dace.to_sdfg(simplify=False)
    ResultE_val = copy.deepcopy(ResultE)
    naive(a_val, b_val, c_val, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, gidx=g, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(U_ref, V_ref, ResultE_ref, U_val, V_val, ResultE_val)

    simple = rollback_dace.to_sdfg(simplify=True)
    ResultE_val = copy.deepcopy(ResultE)
    simple(a_val, b_val, c_val, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, gidx=g, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(U_ref, V_ref, ResultE_ref, U_val, V_val, ResultE_val)

    aopt = copy.deepcopy(simple)
    auto_optimize(aopt, dace.DeviceType.CPU)
    ResultE_val = copy.deepcopy(ResultE)
    aopt(a_val, b_val, c_val, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, gidx=g, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(U_ref, V_ref, ResultE_ref, U_val, V_val, ResultE_val)

    naive_gpu_wcopy = copy.deepcopy(naive)
    naive_gpu_wcopy.apply_gpu_transformations()
    ResultE_val = copy.deepcopy(ResultE)
    naive_gpu_wcopy(a_val, b_val, c_val, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, gidx=g, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(U_ref, V_ref, ResultE_ref, U_val, V_val, ResultE_val)

    simple_gpu_wcopy = copy.deepcopy(simple)
    simple_gpu_wcopy.apply_gpu_transformations()
    ResultE_val = copy.deepcopy(ResultE)
    simple_gpu_wcopy(a_val, b_val, c_val, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, gidx=g, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(U_ref, V_ref, ResultE_ref, U_val, V_val, ResultE_val)

    aopt_gpu_wcopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wcopy, dace.DeviceType.GPU)
    ResultE_val = copy.deepcopy(ResultE)
    aopt_gpu_wcopy(a_val, b_val, c_val, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, gidx=g, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(U_ref, V_ref, ResultE_ref, U_val, V_val, ResultE_val)

    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        print("Cupy not found, skipping GPU w/o HtoD and DtoH copying tests")
        exit(0)
    
    Time = cp.asarray(Time)
    Dx = cp.asarray(Dx)
    Dxx = cp.asarray(Dxx)
    Mux = cp.asarray(Mux)
    Varx = cp.asarray(Varx)
    Dy = cp.asarray(Dy)
    Dyy = cp.asarray(Dyy)
    Muy = cp.asarray(Muy)
    Vary = cp.asarray(Vary)
    ResultE = cp.asarray(ResultE)

    a_val = cp.empty((numXY, numXY), dtype=np.float64)
    b_val = cp.empty((numXY, numXY), dtype=np.float64)
    c_val = cp.empty((numXY, numXY), dtype=np.float64)
    U_val = cp.empty((numY, numX), dtype=np.float64)
    V_val = cp.empty((numX, numY), dtype=np.float64)

    naive_gpu_wocopy = copy.deepcopy(naive)
    apply_gpu_storage(naive_gpu_wocopy)
    naive_gpu_wocopy.apply_gpu_transformations()
    ResultE_val = cp.copy(ResultE)
    naive_gpu_wocopy(a_val, b_val, c_val, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, gidx=g, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(U_ref, V_ref, ResultE_ref, U_val, V_val, ResultE_val)

    simple_gpu_wocopy = copy.deepcopy(simple)
    apply_gpu_storage(simple_gpu_wocopy)
    simple_gpu_wocopy.apply_gpu_transformations()
    ResultE_val = cp.copy(ResultE)
    simple_gpu_wocopy(a_val, b_val, c_val, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, gidx=g, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(U_ref, V_ref, ResultE_ref, U_val, V_val, ResultE_val)

    aopt_gpu_wocopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wocopy, dace.DeviceType.GPU, use_gpu_storage=True)
    ResultE_val = cp.copy(ResultE)
    aopt_gpu_wocopy(a_val, b_val, c_val, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, gidx=g, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(U_ref, V_ref, ResultE_ref, U_val, V_val, ResultE_val)
