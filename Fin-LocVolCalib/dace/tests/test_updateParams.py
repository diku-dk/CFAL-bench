import copy
import dace
import numpy as np
import numpy.typing as npt


from dace.transformation.auto.auto_optimize import apply_gpu_storage, auto_optimize


# NOTE: Is alpha supposed to be used here?
def updateParams_cpython(numX: int, numY: int, g: int, alpha: float, beta: float, nu: float,
                         X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64], Time: npt.NDArray[np.float64],
                         MuX: npt.NDArray[np.float64], VarX: npt.NDArray[np.float64],
                         MuY: npt.NDArray[np.float64], VarY: npt.NDArray[np.float64]):

    # for j in range(numY):
    #     for i in range(numX):
    #         MuX[j, i] = 0.0
    #         VarX[j, i] = np.exp(2 * (beta * np.log(X[i]) + Y[j] - 0.5 * nu * nu * Time[g]))
    MuX[:] = 0.0
    VarX[:] = np.exp(2 * np.add.outer(Y - 0.5 * nu * nu * Time[g], beta * np.log(X)))

    for i in range(numX):
        for j in range(numY):
            MuY[i, j] = 0.0
            VarY[i, j] = nu * nu


g, numX, numY, numT = dace.symbol('g'), dace.symbol('numX'), dace.symbol('numY'), dace.symbol('numT')


@dace.program
def updateParams_dace(alpha: float, beta: float, nu: float,
                      X: dace.float64[numX], Y: dace.float64[numY], Time: dace.float64[numT],
                      MuX: dace.float64[numY, numX], VarX: dace.float64[numY, numX],
                      MuY: dace.float64[numX, numY], VarY: dace.float64[numX, numY]):

    # for j in dace.map[0:numY]:
    #     for i in dace.map[0:numX]:
    #         MuX[j, i] = 0.0
    #         VarX[j, i] = np.exp(2 * (beta * np.log(X[i]) + Y[j] - 0.5 * nu * nu * Time[g]))
    MuX[:] = 0.0
    VarX[:] = np.exp(2 * np.add.outer(Y - 0.5 * nu * nu * Time[g], beta * np.log(X)))

    MuY[:] = 0.0
    VarY[:] = nu * nu


def validate(MuX_ref: npt.NDArray[np.float64], VarX_ref: npt.NDArray[np.float64],
             MuY_ref: npt.NDArray[np.float64], VarY_ref: npt.NDArray[np.float64],
             MuX_val: npt.NDArray[np.float64], VarX_val: npt.NDArray[np.float64],
             MuY_val: npt.NDArray[np.float64], VarY_val: npt.NDArray[np.float64]):
    assert np.allclose(MuX_ref, MuX_val)
    assert np.allclose(VarX_ref, VarX_val)
    assert np.allclose(MuY_ref, MuY_val)
    assert np.allclose(VarY_ref, VarY_val)


if __name__ == "__main__":

    seed = 42
    rng = np.random.default_rng(seed)

    numX = 5000
    numY = 5000
    numT = 5000

    g = rng.integers(0, numT)
    alpha = rng.random()
    beta = rng.random()
    nu = rng.random()
    X = rng.random(numX)
    Y = rng.random(numY)
    Time = rng.random(numT)
    MuX_ref = np.empty((numY, numX), dtype=np.float64)
    VarX_ref = np.empty((numY, numX), dtype=np.float64)
    MuY_ref = np.empty((numX, numY), dtype=np.float64)
    VarY_ref = np.empty((numX, numY), dtype=np.float64)

    updateParams_cpython(numX, numY, g, alpha, beta, nu, X, Y, Time, MuX_ref, VarX_ref, MuY_ref, VarY_ref)

    MuX_val = np.empty((numY, numX), dtype=np.float64)
    VarX_val = np.empty((numY, numX), dtype=np.float64)
    MuY_val = np.empty((numX, numY), dtype=np.float64)
    VarY_val = np.empty((numX, numY), dtype=np.float64)

    naive = updateParams_dace.to_sdfg(simplify=False)
    naive(alpha, beta, nu, X, Y, Time, MuX_val, VarX_val, MuY_val, VarY_val, g=g, numX=numX, numY=numY, numT=numT)
    validate(MuX_ref, VarX_ref, MuY_ref, VarY_ref, MuX_val, VarX_val, MuY_val, VarY_val)

    simple = updateParams_dace.to_sdfg(simplify=True)
    simple(alpha, beta, nu, X, Y, Time, MuX_val, VarX_val, MuY_val, VarY_val, g=g, numX=numX, numY=numY, numT=numT)
    validate(MuX_ref, VarX_ref, MuY_ref, VarY_ref, MuX_val, VarX_val, MuY_val, VarY_val)

    aopt = copy.deepcopy(simple)
    auto_optimize(aopt, dace.DeviceType.CPU)
    aopt(alpha, beta, nu, X, Y, Time, MuX_val, VarX_val, MuY_val, VarY_val, g=g, numX=numX, numY=numY, numT=numT)
    validate(MuX_ref, VarX_ref, MuY_ref, VarY_ref, MuX_val, VarX_val, MuY_val, VarY_val)

    naive_gpu_wcopy = copy.deepcopy(naive)
    naive_gpu_wcopy.apply_gpu_transformations()
    naive_gpu_wcopy(alpha, beta, nu, X, Y, Time, MuX_val, VarX_val, MuY_val, VarY_val, g=g, numX=numX, numY=numY, numT=numT)
    validate(MuX_ref, VarX_ref, MuY_ref, VarY_ref, MuX_val, VarX_val, MuY_val, VarY_val)

    simple_gpu_wcopy = copy.deepcopy(simple)
    simple_gpu_wcopy.apply_gpu_transformations()
    simple_gpu_wcopy(alpha, beta, nu, X, Y, Time, MuX_val, VarX_val, MuY_val, VarY_val, g=g, numX=numX, numY=numY, numT=numT)
    validate(MuX_ref, VarX_ref, MuY_ref, VarY_ref, MuX_val, VarX_val, MuY_val, VarY_val)

    aopt_gpu_wcopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wcopy, dace.DeviceType.GPU)
    aopt_gpu_wcopy(alpha, beta, nu, X, Y, Time, MuX_val, VarX_val, MuY_val, VarY_val, g=g, numX=numX, numY=numY, numT=numT)
    validate(MuX_ref, VarX_ref, MuY_ref, VarY_ref, MuX_val, VarX_val, MuY_val, VarY_val)

    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        print("Cupy not found, skipping GPU w/o HtoD and DtoH copying tests")
        exit(0)

    X_dev = cp.asarray(X)
    Y_dev = cp.asarray(Y)
    Time_dev = cp.asarray(Time)
    MuX_val = cp.empty((numY, numX), dtype=np.float64)
    VarX_val = cp.empty((numY, numX), dtype=np.float64)
    MuY_val = cp.empty((numX, numY), dtype=np.float64)
    VarY_val = cp.empty((numX, numY), dtype=np.float64)

    naive_gpu_wocopy = copy.deepcopy(naive)
    apply_gpu_storage(naive_gpu_wocopy)
    naive_gpu_wocopy.apply_gpu_transformations()
    naive_gpu_wocopy(alpha, beta, nu, X_dev, Y_dev, Time_dev, MuX_val, VarX_val, MuY_val, VarY_val, g=g, numX=numX, numY=numY, numT=numT)
    validate(MuX_ref, VarX_ref, MuY_ref, VarY_ref, MuX_val, VarX_val, MuY_val, VarY_val)

    simple_gpu_wocopy = copy.deepcopy(simple)
    apply_gpu_storage(simple_gpu_wocopy)
    simple_gpu_wocopy.apply_gpu_transformations()
    simple_gpu_wocopy(alpha, beta, nu, X_dev, Y_dev, Time_dev, MuX_val, VarX_val, MuY_val, VarY_val, g=g, numX=numX, numY=numY, numT=numT)
    validate(MuX_ref, VarX_ref, MuY_ref, VarY_ref, MuX_val, VarX_val, MuY_val, VarY_val)

    aopt_gpu_wocopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wocopy, dace.DeviceType.GPU, use_gpu_storage=True)
    aopt_gpu_wocopy(alpha, beta, nu, X_dev, Y_dev, Time_dev, MuX_val, VarX_val, MuY_val, VarY_val, g=g, numX=numX, numY=numY, numT=numT)
    validate(MuX_ref, VarX_ref, MuY_ref, VarY_ref, MuX_val, VarX_val, MuY_val, VarY_val)
    