import copy
import dace
import numpy as np
import numpy.typing as npt


from dace.transformation.auto.auto_optimize import apply_gpu_storage, auto_optimize


def initGrid_cpython(numX: int, numY: int, numT: int,
                     s0: float, alpha: float, nu: float, t: float,
                     X: npt.NDArray[np.float64],
                     Y: npt.NDArray[np.float64],
                     Time: npt.NDArray[np.float64]) -> tuple[int, int]:
    """ Initializes indX, indY, X, Y, and Time. """

    for i in range(numT):
        Time[i] = t * i / (numT - 1)

    stdX = 20 * alpha * s0 * np.sqrt(t)
    dx = stdX / numX
    indX = int(s0 / dx)

    for i in range(numX):
        X[i] = i * dx - indX * dx + s0

    stdY = 10 * nu * np.sqrt(t)
    dy = stdY / numY
    logAlpha = np.log(alpha)
    indY = int(numY / 2)

    for i in range(numY):
        Y[i] = i * dy - indY * dy + logAlpha
    return indX, indY


numX, numY, numT = dace.symbol('numX'), dace.symbol('numY'), dace.symbol('numT')


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


def validate(X_ref: npt.NDArray[np.float64],
             Y_ref: npt.NDArray[np.float64],
             Time_ref: npt.NDArray[np.float64],
             X_val: npt.NDArray[np.float64],
             Y_val: npt.NDArray[np.float64],
             Time_val: npt.NDArray[np.float64],
             indX_ref: int,
             indY_ref: int,
             indX_val: int,
             indY_val: int):
    assert indX_ref == indX_val and indY_ref == indY_val
    assert np.allclose(Time_ref, Time_val)
    assert np.allclose(X_ref, X_val)
    assert np.allclose(Y_ref, Y_val)


if __name__ == "__main__":

    seed = 42
    rng = np.random.default_rng(seed)

    numT = 1000000
    numX = 1000000
    numY = 1000000

    s0 = rng.random()
    alpha = rng.random()
    nu = rng.random()

    Time_ref = np.empty(numT, dtype=np.float64)
    X_ref = np.empty(numX, dtype=np.float64)
    Y_ref = np.empty(numY, dtype=np.float64)

    indX_ref, indY_ref = initGrid_cpython(numX, numY, numT, s0, alpha, nu, 1.0, X_ref, Y_ref, Time_ref)

    Time_val = np.empty(numT, dtype=np.float64)
    X_val = np.empty(numX, dtype=np.float64)
    Y_val = np.empty(numY, dtype=np.float64)

    naive = initGrid_dace.to_sdfg(simplify=False)
    indX_val, indY_val = naive(s0, alpha, nu, 1.0, X_val, Y_val, Time_val, numX=numX, numY=numY, numT=numT)
    validate(X_ref, Y_ref, Time_ref, X_val, Y_val, Time_val, indX_ref, indY_ref, indX_val, indY_val)

    simple = initGrid_dace.to_sdfg(simplify=True)
    indX_val, indY_val = simple(s0, alpha, nu, 1.0, X_val, Y_val, Time_val, numX=numX, numY=numY, numT=numT)
    validate(X_ref, Y_ref, Time_ref, X_val, Y_val, Time_val, indX_ref, indY_ref, indX_val, indY_val)

    aopt = copy.deepcopy(simple)
    auto_optimize(aopt, dace.DeviceType.CPU)
    indX_val, indY_val = aopt(s0, alpha, nu, 1.0, X_val, Y_val, Time_val, numX=numX, numY=numY, numT=numT)
    validate(X_ref, Y_ref, Time_ref, X_val, Y_val, Time_val, indX_ref, indY_ref, indX_val, indY_val)

    naive_gpu_wcopy = copy.deepcopy(naive)
    naive_gpu_wcopy.apply_gpu_transformations()
    indX_val, indY_val = naive_gpu_wcopy(s0, alpha, nu, 1.0, X_val, Y_val, Time_val, numX=numX, numY=numY, numT=numT)
    validate(X_ref, Y_ref, Time_ref, X_val, Y_val, Time_val, indX_ref, indY_ref, indX_val, indY_val)

    simple_gpu_wcopy = copy.deepcopy(simple)
    simple_gpu_wcopy.apply_gpu_transformations()
    indX_val, indY_val = simple_gpu_wcopy(s0, alpha, nu, 1.0, X_val, Y_val, Time_val, numX=numX, numY=numY, numT=numT)
    validate(X_ref, Y_ref, Time_ref, X_val, Y_val, Time_val, indX_ref, indY_ref, indX_val, indY_val)

    aopt_gpu_wcopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wcopy, dace.DeviceType.GPU)
    indX_val, indY_val = aopt_gpu_wcopy(s0, alpha, nu, 1.0, X_val, Y_val, Time_val, numX=numX, numY=numY, numT=numT)
    validate(X_ref, Y_ref, Time_ref, X_val, Y_val, Time_val, indX_ref, indY_ref, indX_val, indY_val)

    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        print("Cupy not found, skipping GPU w/o HtoD and DtoH copying tests")
        exit(0)

    Time_val = cp.empty(numT, dtype=np.float64)
    X_val = cp.empty(numX, dtype=np.float64)
    Y_val = cp.empty(numY, dtype=np.float64)

    naive_gpu_wocopy = copy.deepcopy(naive)
    apply_gpu_storage(naive_gpu_wocopy)
    naive_gpu_wocopy.apply_gpu_transformations()
    indX_val, indY_val = naive_gpu_wocopy(s0, alpha, nu, 1.0, X_val, Y_val, Time_val, numX=numX, numY=numY, numT=numT)
    validate(X_ref, Y_ref, Time_ref, X_val, Y_val, Time_val, indX_ref, indY_ref, indX_val, indY_val)

    simple_gpu_wocopy = copy.deepcopy(simple)
    apply_gpu_storage(simple_gpu_wocopy)
    simple_gpu_wocopy.apply_gpu_transformations()
    indX_val, indY_val = simple_gpu_wocopy(s0, alpha, nu, 1.0, X_val, Y_val, Time_val, numX=numX, numY=numY, numT=numT)
    validate(X_ref, Y_ref, Time_ref, X_val, Y_val, Time_val, indX_ref, indY_ref, indX_val, indY_val)

    aopt_gpu_wocopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wocopy, dace.DeviceType.GPU, use_gpu_storage=True)
    indX_val, indY_val = aopt_gpu_wocopy(s0, alpha, nu, 1.0, X_val, Y_val, Time_val, numX=numX, numY=numY, numT=numT)
    validate(X_ref, Y_ref, Time_ref, X_val, Y_val, Time_val, indX_ref, indY_ref, indX_val, indY_val)
    


