import copy
import dace
import numpy as np
import numpy.typing as npt


from dace.transformation.auto.auto_optimize import apply_gpu_storage, auto_optimize


def initOperator_cpython(N: int, xx: npt.NDArray[np.float64], D: npt.NDArray[np.float64], DD: npt.NDArray[np.float64]):
    """ Initializes Globals Dx[Dy] and Dxx[Dyy]. """

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


N = dace.symbol('N')


@dace.program
def initOperator_dace(xx: dace.float64[N], D: dace.float64[N, 3], DD: dace.float64[N,3]):
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
    for i in dace.map[1:N - 1]:
        dxl = xx[i] - xx[i - 1]
        dxu = xx[i + 1] - xx[i]

        D[i, 0] = -dxu / dxl / (dxl + dxu)
        D[i, 1] = (dxu / dxl - dxl / dxu) / (dxl + dxu)
        D[i, 2] = dxl / dxu / (dxl + dxu)

        DD[i, 0] = 2.0 / dxl / (dxl + dxu)
        DD[i, 1] = -2.0 * (1.0 / dxl + 1.0 / dxu) / (dxl + dxu)
        DD[i, 2] = 2.0 / dxu / (dxl + dxu)

    # upper boundary
    dxlu = xx[N - 1] - xx[N - 2]
    dxuu = 0.0

    D[(N - 1), 0] = -1.0 / dxlu
    D[(N - 1), 1] = 1.0 / dxlu
    D[(N - 1), 2] = 0.0

    DD[(N - 1), 0] = 0.0
    DD[(N - 1), 1] = 0.0
    DD[(N - 1), 2] = 0.0


def validate(D_ref: npt.NDArray[np.float64], DD_ref: npt.NDArray[np.float64],
             D_val: npt.NDArray[np.float64], DD_val: npt.NDArray[np.float64]):
    assert np.allclose(D_ref, D_val)
    assert np.allclose(DD_ref, DD_val)


if __name__ == "__main__":

    seed = 42
    rng = np.random.default_rng(seed)

    N = 1000000

    xx = rng.random(N)
    D_ref = np.empty((N, 3), dtype=np.float64)
    DD_ref = np.empty((N, 3), dtype=np.float64)

    initOperator_cpython(N, xx, D_ref, DD_ref)

    D_val = np.empty((N, 3), dtype=np.float64)
    DD_val = np.empty((N, 3), dtype=np.float64)

    naive = initOperator_dace.to_sdfg(simplify=False)
    naive(xx, D_val, DD_val, N=N)
    validate(D_ref, DD_ref, D_val, DD_val)

    simple = initOperator_dace.to_sdfg(simplify=True)
    simple(xx, D_val, DD_val, N=N)
    validate(D_ref, DD_ref, D_val, DD_val)

    aopt = copy.deepcopy(simple)
    auto_optimize(aopt, dace.DeviceType.CPU)
    aopt(xx, D_val, DD_val, N=N)
    validate(D_ref, DD_ref, D_val, DD_val)

    naive_gpu_wcopy = copy.deepcopy(naive)
    naive_gpu_wcopy.apply_gpu_transformations()
    naive_gpu_wcopy(xx, D_val, DD_val, N=N)
    validate(D_ref, DD_ref, D_val, DD_val)

    simple_gpu_wcopy = copy.deepcopy(simple)
    simple_gpu_wcopy.apply_gpu_transformations()
    simple_gpu_wcopy(xx, D_val, DD_val, N=N)
    validate(D_ref, DD_ref, D_val, DD_val)

    aopt_gpu_wcopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wcopy, dace.DeviceType.GPU)
    aopt_gpu_wcopy(xx, D_val, DD_val, N=N)
    validate(D_ref, DD_ref, D_val, DD_val)

    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        print("Cupy not found, skipping GPU w/o HtoD and DtoH copying tests")
        exit(0)

    xx_dev = cp.asarray(xx)
    D_val = cp.empty((N, 3), dtype=np.float64)
    DD_val = cp.empty((N, 3), dtype=np.float64)

    naive_gpu_wocopy = copy.deepcopy(naive)
    apply_gpu_storage(naive_gpu_wocopy)
    naive_gpu_wocopy.apply_gpu_transformations()
    naive_gpu_wocopy(xx_dev, D_val, DD_val, N=N)
    validate(D_ref, DD_ref, D_val, DD_val)

    simple_gpu_wocopy = copy.deepcopy(simple)
    apply_gpu_storage(simple_gpu_wocopy)
    simple_gpu_wocopy.apply_gpu_transformations()
    simple_gpu_wocopy(xx_dev, D_val, DD_val, N=N)
    validate(D_ref, DD_ref, D_val, DD_val)

    aopt_gpu_wocopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wocopy, dace.DeviceType.GPU, use_gpu_storage=True)
    aopt_gpu_wocopy(xx_dev, D_val, DD_val, N=N)
    validate(D_ref, DD_ref, D_val, DD_val)
    