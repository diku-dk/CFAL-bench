import copy
import dace
import numpy as np
import numpy.typing as npt


from dace.transformation.auto.auto_optimize import apply_gpu_storage, auto_optimize


def tridiag_cpython(N: int,
                    a: npt.NDArray[np.float64],
                    b: npt.NDArray[np.float64],
                    c: npt.NDArray[np.float64],
                    y: npt.NDArray[np.float64]):
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


N = dace.symbol('N')


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


def validate(b_ref: npt.NDArray[np.float64], y_ref: npt.NDArray[np.float64],
             b_val: npt.NDArray[np.float64], y_val: npt.NDArray[np.float64]):
    assert np.allclose(b_ref, b_val)
    assert np.allclose(y_ref, y_val)


if __name__ == "__main__":

    seed = 42
    rng = np.random.default_rng(seed)

    N = 100000

    a = rng.random(N)
    b = rng.random(N)
    c = rng.random(N)
    y = rng.random(N)

    # # Make sure the matrix is diagonally dominant
    # a[0] += abs(c[0])
    # for i in range(1, N - 1):
    #     a[i] += abs(b[i - 1] + c[i])
    # a[-1] += abs(b[-2])

    b_ref = np.copy(b)
    y_ref = np.copy(y)
    tridiag_cpython(N, a, b_ref, c, y_ref)
    
    naive = tridiag_dace.to_sdfg(simplify=False)
    b_val = np.copy(b)
    y_val = np.copy(y)
    naive(a, b_val, c, y_val, N=N)
    validate(b_ref, y_ref, b_val, y_val)

    simple = tridiag_dace.to_sdfg(simplify=True)
    b_val = np.copy(b)
    y_val = np.copy(y)
    simple(a, b_val, c, y_val, N=N)
    validate(b_ref, y_ref, b_val, y_val)

    aopt = copy.deepcopy(simple)
    auto_optimize(aopt, dace.DeviceType.CPU)
    b_val = np.copy(b)
    y_val = np.copy(y)
    aopt(a, b_val, c, y_val, N=N)
    validate(b_ref, y_ref, b_val, y_val)

    naive_gpu_wcopy = copy.deepcopy(naive)
    naive_gpu_wcopy.apply_gpu_transformations()
    b_val = np.copy(b)
    y_val = np.copy(y)
    naive_gpu_wcopy(a, b_val, c, y_val, N=N)
    validate(b_ref, y_ref, b_val, y_val)

    simple_gpu_wcopy = copy.deepcopy(simple)
    simple_gpu_wcopy.apply_gpu_transformations()
    b_val = np.copy(b)
    y_val = np.copy(y)
    simple_gpu_wcopy(a, b_val, c, y_val, N=N)
    validate(b_ref, y_ref, b_val, y_val)

    aopt_gpu_wcopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wcopy, dace.DeviceType.GPU)
    b_val = np.copy(b)
    y_val = np.copy(y)
    aopt_gpu_wcopy(a, b_val, c, y_val, N=N)
    validate(b_ref, y_ref, b_val, y_val)

    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        print("Cupy not found, skipping GPU w/o HtoD and DtoH copying tests")
        exit(0)
    
    a = cp.asarray(a)
    b = cp.asarray(b)
    c = cp.asarray(c)
    y = cp.asarray(y)

    naive_gpu_wocopy = copy.deepcopy(naive)
    apply_gpu_storage(naive_gpu_wocopy)
    naive_gpu_wocopy.apply_gpu_transformations()
    b_val = cp.copy(b)
    y_val = cp.copy(y)
    naive_gpu_wocopy(a, b_val, c, y_val, N=N)
    validate(b_ref, y_ref, b_val, y_val)

    simple_gpu_wocopy = copy.deepcopy(simple)
    apply_gpu_storage(simple_gpu_wocopy)
    simple_gpu_wocopy.apply_gpu_transformations()
    b_val = cp.copy(b)
    y_val = cp.copy(y)
    simple_gpu_wocopy(a, b_val, c, y_val, N=N)
    validate(b_ref, y_ref, b_val, y_val)

    aopt_gpu_wocopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wocopy, dace.DeviceType.GPU, use_gpu_storage=True)
    b_val = cp.copy(b)
    y_val = cp.copy(y)
    aopt_gpu_wocopy(a, b_val, c, y_val, N=N)
    validate(b_ref, y_ref, b_val, y_val)

