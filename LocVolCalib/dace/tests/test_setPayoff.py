import copy
import dace
import numpy as np
import numpy.typing as npt


from dace.transformation.auto.auto_optimize import apply_gpu_storage, auto_optimize


def setPayoff_cpython(numX: int, numY: int, strike: float,
                      X: npt.NDArray[np.float64],
                      ResultE: npt.NDArray[np.float64]):

    # for i in range(numX):
    #     payoff = max(X[i] - strike, 0.0)

    #     for j in range(numY):
    #         ResultE[i, j] = payoff
    ResultE[:] = np.reshape(np.maximum(X - strike, 0.0), (numX, 1))


numX, numY = dace.symbol('numX'), dace.symbol('numY')


@dace.program
def setPayoff_dace(strike: float, X: dace.float64[numX], ResultE: dace.float64[numX, numY]):

    # for i in dace.map[0:numX]:
    #     payoff = max(X[i] - strike, 0.0)
    #     ResultE[i] = payoff
    ResultE[:] = np.reshape(np.maximum(X - strike, 0.0), (numX, 1))


def validate(ResultE_ref: npt.NDArray[np.float64], ResultE_val: npt.NDArray[np.float64]):
    assert np.allclose(ResultE_ref, ResultE_val)


if __name__ == "__main__":

    seed = 42
    rng = np.random.default_rng(seed)

    numX = 10000
    numY = 10000

    strike = rng.random()
    X = rng.random(numX)
    ResultE_ref = np.empty((numX, numY), dtype=np.float64)

    setPayoff_cpython(numX, numY, strike, X, ResultE_ref)

    ResultE_val = np.empty((numX, numY), dtype=np.float64)

    naive = setPayoff_dace.to_sdfg(simplify=False)
    naive(strike, X, ResultE_val, numX=numX, numY=numY)
    validate(ResultE_ref, ResultE_val)

    simple = setPayoff_dace.to_sdfg(simplify=True)
    simple(strike, X, ResultE_val, numX=numX, numY=numY)
    validate(ResultE_ref, ResultE_val)

    aopt = copy.deepcopy(simple)
    auto_optimize(aopt, dace.DeviceType.CPU)
    aopt(strike, X, ResultE_val, numX=numX, numY=numY)
    validate(ResultE_ref, ResultE_val)

    naive_gpu_wcopy = copy.deepcopy(naive)
    naive_gpu_wcopy.apply_gpu_transformations()
    naive_gpu_wcopy(strike, X, ResultE_val, numX=numX, numY=numY)
    validate(ResultE_ref, ResultE_val)

    simple_gpu_wcopy = copy.deepcopy(simple)
    simple_gpu_wcopy.apply_gpu_transformations()
    simple_gpu_wcopy(strike, X, ResultE_val, numX=numX, numY=numY)
    validate(ResultE_ref, ResultE_val)

    aopt_gpu_wcopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wcopy, dace.DeviceType.GPU)
    aopt_gpu_wcopy(strike, X, ResultE_val, numX=numX, numY=numY)
    validate(ResultE_ref, ResultE_val)

    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        print("Cupy not found, skipping GPU w/o HtoD and DtoH copying tests")
        exit(0)

    X_dev = cp.asarray(X)
    ResultE_val = cp.empty((numX, numY), dtype=np.float64)

    naive_gpu_wocopy = copy.deepcopy(naive)
    apply_gpu_storage(naive_gpu_wocopy)
    naive_gpu_wocopy.apply_gpu_transformations()
    naive_gpu_wocopy(strike, X_dev, ResultE_val, numX=numX, numY=numY)
    validate(ResultE_ref, ResultE_val)

    simple_gpu_wocopy = copy.deepcopy(simple)
    apply_gpu_storage(simple_gpu_wocopy)
    simple_gpu_wocopy.apply_gpu_transformations()
    simple_gpu_wocopy(strike, X_dev, ResultE_val, numX=numX, numY=numY)
    validate(ResultE_ref, ResultE_val)

    aopt_gpu_wocopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wocopy, dace.DeviceType.GPU, use_gpu_storage=True)
    aopt_gpu_wocopy(strike, X_dev, ResultE_val, numX=numX, numY=numY)
    validate(ResultE_ref, ResultE_val)
    