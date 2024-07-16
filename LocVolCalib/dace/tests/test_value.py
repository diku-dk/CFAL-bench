import copy
import dace
import numpy as np
import numpy.typing as npt


from test_setPayoff import setPayoff_cpython, setPayoff_dace
from test_updateParams import updateParams_cpython, updateParams_dace
from test_rollback import rollback_cpython, rollback_dace
from dace.transformation.auto.auto_optimize import apply_gpu_storage, auto_optimize


def value_cpython(strike: float, alpha: float, beta: float, nu: float,
                  numX: int, numY: int, numT: int, indX: int, indY: int,
                  a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], c: npt.NDArray[np.float64],
                  X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64], Time: npt.NDArray[np.float64],
                  U: npt.NDArray[np.float64], V: npt.NDArray[np.float64],
                  Dx: npt.NDArray[np.float64], Dxx: npt.NDArray[np.float64],
                  MuX: npt.NDArray[np.float64], VarX: npt.NDArray[np.float64],
                  Dy: npt.NDArray[np.float64], Dyy: npt.NDArray[np.float64],
                  MuY: npt.NDArray[np.float64], VarY: npt.NDArray[np.float64],
                  ResultE: npt.NDArray[np.float64]):

    setPayoff_cpython(numX, numY, strike, X, ResultE)
    for i in range(numT - 2, -1, -1):
        updateParams_cpython(numX, numY, i, alpha, beta, nu, X, Y, Time, MuX, VarX, MuY, VarY)
        rollback_cpython(numX, numY, i, a, b, c, Time, U, V, Dx, Dxx, MuX, VarX, Dy, Dyy, MuY, VarY, ResultE)

    res = ResultE[indX, indY]
    return res


numXY, numX, numY, numT = dace.symbol('numXY'), dace.symbol('numX'), dace.symbol('numY'), dace.symbol('numT')
indX, indY = dace.symbol('indX'), dace.symbol('indY')


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
    
    res = ResultE[indX, indY]
    return res


def validate(ResultE_ref: npt.NDArray[np.float64], ResultE_val: npt.NDArray[np.float64]):
    assert np.allclose(ResultE_ref, ResultE_val)


if __name__ == "__main__":

    seed = 42
    rng = np.random.default_rng(seed)

    numX = 256
    numY = 256
    numT = 32
    numXY = max(numX, numY)

    indX = rng.integers(0, numX)
    indY = rng.integers(0, numY)
    strike = rng.random()
    alpha = rng.random()
    beta = rng.random()
    nu = rng.random()
    X = rng.random(numX)
    Y = rng.random(numY)
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
    value_cpython(strike, alpha, beta, nu, numX, numY, numT, indX, indY, a_ref, b_ref, c_ref, X, Y, Time, U_ref, V_ref, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_ref)

    a_val = np.empty((numXY, numXY), dtype=np.float64)
    b_val = np.empty((numXY, numXY), dtype=np.float64)
    c_val = np.empty((numXY, numXY), dtype=np.float64)
    U_val = np.empty((numY, numX), dtype=np.float64)
    V_val = np.empty((numX, numY), dtype=np.float64)

    naive = value_dace.to_sdfg(simplify=False)
    ResultE_val = copy.deepcopy(ResultE)
    naive(strike, alpha, beta, nu, a_val, b_val, c_val, X, Y, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, numX=numX, numY=numY, numT=numT, numXY=numXY, indX=indX, indY=indY)
    validate(ResultE_ref, ResultE_val)

    simple = value_dace.to_sdfg(simplify=True)
    ResultE_val = copy.deepcopy(ResultE)
    simple(strike, alpha, beta, nu, a_val, b_val, c_val, X, Y, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, numX=numX, numY=numY, numT=numT, numXY=numXY, indX=indX, indY=indY)
    validate(ResultE_ref, ResultE_val)

    aopt = copy.deepcopy(simple)
    auto_optimize(aopt, dace.DeviceType.CPU)
    ResultE_val = copy.deepcopy(ResultE)
    aopt(strike, alpha, beta, nu, a_val, b_val, c_val, X, Y, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, numX=numX, numY=numY, numT=numT, numXY=numXY, indX=indX, indY=indY)
    validate(ResultE_ref, ResultE_val)

    naive_gpu_wcopy = copy.deepcopy(naive)
    naive_gpu_wcopy.apply_gpu_transformations()
    ResultE_val = copy.deepcopy(ResultE)
    naive_gpu_wcopy(strike, alpha, beta, nu, a_val, b_val, c_val, X, Y, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, numX=numX, numY=numY, numT=numT, numXY=numXY, indX=indX, indY=indY)
    validate(ResultE_ref, ResultE_val)

    simple_gpu_wcopy = copy.deepcopy(simple)
    simple_gpu_wcopy.apply_gpu_transformations()
    ResultE_val = copy.deepcopy(ResultE)
    simple_gpu_wcopy(strike, alpha, beta, nu, a_val, b_val, c_val, X, Y, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, numX=numX, numY=numY, numT=numT, numXY=numXY, indX=indX, indY=indY)
    validate(ResultE_ref, ResultE_val)

    aopt_gpu_wcopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wcopy, dace.DeviceType.GPU)
    ResultE_val = copy.deepcopy(ResultE)
    aopt_gpu_wcopy(strike, alpha, beta, nu, a_val, b_val, c_val, X, Y, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, numX=numX, numY=numY, numT=numT, numXY=numXY, indX=indX, indY=indY)
    
    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        print("Cupy not found, skipping GPU w/o HtoD and DtoH copying tests")
        exit(0)

    X = cp.asarray(X)
    Y = cp.asarray(Y)
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
    naive_gpu_wocopy(strike, alpha, beta, nu, a_val, b_val, c_val, X, Y, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, numX=numX, numY=numY, numT=numT, numXY=numXY, indX=indX, indY=indY)
    validate(ResultE_ref, ResultE_val)

    simple_gpu_wocopy = copy.deepcopy(simple)
    apply_gpu_storage(simple_gpu_wocopy)
    simple_gpu_wocopy.apply_gpu_transformations()
    ResultE_val = cp.copy(ResultE)
    simple_gpu_wocopy(strike, alpha, beta, nu, a_val, b_val, c_val, X, Y, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, numX=numX, numY=numY, numT=numT, numXY=numXY, indX=indX, indY=indY)
    validate(ResultE_ref, ResultE_val)

    aopt_gpu_wocopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wocopy, dace.DeviceType.GPU, use_gpu_storage=True)
    ResultE_val = cp.copy(ResultE)
    aopt_gpu_wocopy(strike, alpha, beta, nu, a_val, b_val, c_val, X, Y, Time, U_val, V_val, Dx, Dxx, Mux, Varx, Dy, Dyy, Muy, Vary, ResultE_val, numX=numX, numY=numY, numT=numT, numXY=numXY, indX=indX, indY=indY)
    validate(ResultE_ref, ResultE_val)
