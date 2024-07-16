import copy
import dace
import numpy as np
import numpy.typing as npt


from test_initGrid import initGrid_cpython, initGrid_dace
from test_initOperator import initOperator_cpython, initOperator_dace
from test_value import value_cpython, value_dace
from dace.transformation.auto.auto_optimize import apply_gpu_storage, auto_optimize


def LocVolCalib_cpython(outer: int, numX: int, numY: int, numT: int,
                        s0: float, alpha: float, beta: float, nu: float, t: float,
                        a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], c: npt.NDArray[np.float64],
                        X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64], Time: npt.NDArray[np.float64],
                        U: npt.NDArray[np.float64], V: npt.NDArray[np.float64],
                        Dx: npt.NDArray[np.float64], Dxx: npt.NDArray[np.float64],
                        MuX: npt.NDArray[np.float64], VarX: npt.NDArray[np.float64],
                        Dy: npt.NDArray[np.float64], Dyy: npt.NDArray[np.float64],
                        MuY: npt.NDArray[np.float64], VarY: npt.NDArray[np.float64],
                        ResultE: npt.NDArray[np.float64], result: npt.NDArray[np.float64]) -> float:

    indX, indY = initGrid_cpython(numX, numY, numT, s0, alpha, nu, t, X, Y, Time)
    initOperator_cpython(numX, X, Dx, Dxx)
    initOperator_cpython(numY, Y, Dy, Dyy)
    for i in range(outer):
        strike = 0.001 * i
        result[i] = value_cpython(strike, alpha, beta, nu, numX, numY, numT, indX, indY,
                                  a[i], b[i], c[i], X, Y, Time,
                                  U[i], V[i], Dx, Dxx, MuX[i], VarX[i], Dy, Dyy, MuY[i], VarY[i], ResultE[i])
    return


outer, numXY, numX, numY, numT = dace.symbol('outer'), dace.symbol('numXY'), dace.symbol('numX'), dace.symbol('numY'), dace.symbol('numT')


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
        result[io] = value_dace(strike, alpha, beta, nu, a[io], b[io], c[io], X, Y, Time,
                                U[io], V[io], Dx, Dxx, MuX[io], VarX[io], Dy, Dyy, MuY[io], VarY[io],
                                ResultE[io], indX=indX, indY=indY)


def validate(result_ref: npt.NDArray[np.float64], result_val: npt.NDArray[np.float64]):
    assert np.allclose(result_ref, result_val)


if __name__ == "__main__":

    seed = 42
    rng = np.random.default_rng(seed)

    outer = 16
    numX = 32
    numY = 256
    numT = 256
    numXY = max(numX, numY)

    # s0 = rng.random()
    # alpha = rng.random()
    # beta = rng.random()
    # nu = rng.random()
    # t = rng.random()
    s0 = 0.03
    t = 5.0
    alpha = 0.2
    nu = 0.6
    beta = 0.5

    a_ref = np.empty((outer, numXY, numXY), dtype=np.float64)
    b_ref = np.empty((outer, numXY, numXY), dtype=np.float64)
    c_ref = np.empty((outer, numXY, numXY), dtype=np.float64)
    X_ref = np.empty(numX, dtype=np.float64)
    Y_ref = np.empty(numY, dtype=np.float64)
    Time_ref = np.empty(numT, dtype=np.float64)
    U_ref = np.empty((outer, numY, numX), dtype=np.float64)
    V_ref = np.empty((outer, numX, numY), dtype=np.float64)
    Dx_ref = np.empty((numX, 3), dtype=np.float64)
    Dxx_ref = np.empty((numX, 3), dtype=np.float64)
    MuX_ref = np.empty((outer, numY, numX), dtype=np.float64)
    VarX_ref = np.empty((outer, numY, numX), dtype=np.float64)
    Dy_ref = np.empty((numY, 3), dtype=np.float64)
    Dyy_ref = np.empty((numY, 3), dtype=np.float64)
    MuY_ref = np.empty((outer, numX, numY), dtype=np.float64)
    VarY_ref = np.empty((outer, numX, numY), dtype=np.float64)
    ResultE_ref = np.empty((outer, numX, numY), dtype=np.float64)
    result_ref = np.empty(outer, dtype=np.float64)

    LocVolCalib_cpython(outer, numX, numY, numT, s0, alpha, beta, nu, t,
                        a_ref, b_ref, c_ref, X_ref, Y_ref, Time_ref, U_ref, V_ref,
                        Dx_ref, Dxx_ref, MuX_ref, VarX_ref, Dy_ref, Dyy_ref, MuY_ref, VarY_ref, ResultE_ref, result_ref)
    
    a_val = np.empty((outer, numXY, numXY), dtype=np.float64)
    b_val = np.empty((outer, numXY, numXY), dtype=np.float64)
    c_val = np.empty((outer, numXY, numXY), dtype=np.float64)
    X_val = np.empty(numX, dtype=np.float64)
    Y_val = np.empty(numY, dtype=np.float64)
    Time_val = np.empty(numT, dtype=np.float64)
    U_val = np.empty((outer, numY, numX), dtype=np.float64)
    V_val = np.empty((outer, numX, numY), dtype=np.float64)
    Dx_val = np.empty((numX, 3), dtype=np.float64)
    Dxx_val = np.empty((numX, 3), dtype=np.float64)
    MuX_val = np.empty((outer, numY, numX), dtype=np.float64)
    VarX_val = np.empty((outer, numY, numX), dtype=np.float64)
    Dy_val = np.empty((numY, 3), dtype=np.float64)
    Dyy_val = np.empty((numY, 3), dtype=np.float64)
    MuY_val = np.empty((outer, numX, numY), dtype=np.float64)
    VarY_val = np.empty((outer, numX, numY), dtype=np.float64)

    naive = LocVolCalib_dace.to_sdfg(simplify=False)
    ResultE_val = np.empty((outer, numX, numY), dtype=np.float64)
    result_val = np.empty(outer, dtype=np.float64)
    naive(s0, alpha, beta, nu, t, a_val, b_val, c_val, X_val, Y_val, Time_val, U_val, V_val, Dx_val, Dxx_val, MuX_val, VarX_val, Dy_val, Dyy_val, MuY_val, VarY_val, ResultE_val, result_val, outer=outer, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(result_ref, result_val)

    simple = copy.deepcopy(naive)
    simple.simplify()
    ResultE_val = np.empty((outer, numX, numY), dtype=np.float64)
    result_val = np.empty(outer, dtype=np.float64)
    simple(s0, alpha, beta, nu, t, a_val, b_val, c_val, X_val, Y_val, Time_val, U_val, V_val, Dx_val, Dxx_val, MuX_val, VarX_val, Dy_val, Dyy_val, MuY_val, VarY_val, ResultE_val, result_val, outer=outer, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(result_ref, result_val)

    aopt = copy.deepcopy(simple)
    auto_optimize(aopt, dace.DeviceType.CPU)
    ResultE_val = np.empty((outer, numX, numY), dtype=np.float64)
    result_val = np.empty(outer, dtype=np.float64)
    aopt(s0, alpha, beta, nu, t, a_val, b_val, c_val, X_val, Y_val, Time_val, U_val, V_val, Dx_val, Dxx_val, MuX_val, VarX_val, Dy_val, Dyy_val, MuY_val, VarY_val, ResultE_val, result_val, outer=outer, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(result_ref, result_val)

    naive_gpu_wcopy = copy.deepcopy(naive)
    naive_gpu_wcopy.apply_gpu_transformations()
    ResultE_val = np.empty((outer, numX, numY), dtype=np.float64)
    result_val = np.empty(outer, dtype=np.float64)
    naive_gpu_wcopy(s0, alpha, beta, nu, t, a_val, b_val, c_val, X_val, Y_val, Time_val, U_val, V_val, Dx_val, Dxx_val, MuX_val, VarX_val, Dy_val, Dyy_val, MuY_val, VarY_val, ResultE_val, result_val, outer=outer, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(result_ref, result_val)

    simple_gpu_wcopy = copy.deepcopy(simple)
    simple_gpu_wcopy.apply_gpu_transformations()
    ResultE_val = np.empty((outer, numX, numY), dtype=np.float64)
    result_val = np.empty(outer, dtype=np.float64)
    simple_gpu_wcopy(s0, alpha, beta, nu, t, a_val, b_val, c_val, X_val, Y_val, Time_val, U_val, V_val, Dx_val, Dxx_val, MuX_val, VarX_val, Dy_val, Dyy_val, MuY_val, VarY_val, ResultE_val, result_val, outer=outer, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(result_ref, result_val)

    aopt_gpu_wcopy = copy.deepcopy(simple)
    auto_optimize(aopt_gpu_wcopy, dace.DeviceType.GPU)
    ResultE_val = np.empty((outer, numX, numY), dtype=np.float64)
    result_val = np.empty(outer, dtype=np.float64)
    aopt_gpu_wcopy(s0, alpha, beta, nu, t, a_val, b_val, c_val, X_val, Y_val, Time_val, U_val, V_val, Dx_val, Dxx_val, MuX_val, VarX_val, Dy_val, Dyy_val, MuY_val, VarY_val, ResultE_val, result_val, outer=outer, numXY=numXY, numX=numX, numY=numY, numT=numT)
    validate(result_ref, result_val)
