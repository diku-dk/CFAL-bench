"""
    Collections of utils for LocVolCalib benchmark
"""
import constants


def is_pow2(n):
    return (n & (n - 1) == 0) and n != 0


def readDataSet():
    """
    Reads formatted input.
    The current format contains 4 unsigned integer numbers and 5 float numbers, new lines separated
    - OUTER -- no restrictions
    - NUM_X -- 2^x, where x \in {5, ..., 9}
    - NUM_Y -- 2^x, where x \in {5, ..., 9}
    - NUM_T -- no restrictions
    - s0 
    - t 
    - alpha 
    - nu 
    - beta
    """

    outer = int(input("Outer: "))
    num_x = int(input("Num_X: "))
    num_y = int(input("Num_Y: "))
    num_t = int(input("Num_T: "))
    s0 = float(input("s0: "))
    t = float(input("t: "))
    alpha = float(input("alpha: "))
    nu = float(input("nu: "))
    beta = float(input("beta: "))

    # Checks
    assert outer > 0

    assert num_x > 0 and num_x <= constants.WORKGROUP_SIZE and is_pow2(num_x), "Illegal NUM_X value!"

    assert num_y > 0 and num_y <= constants.WORKGROUP_SIZE and is_pow2(num_y), "Illegal NUM_Y value!"

    assert num_t > 0, "NUM_T value less or equal to zero!!"

    print(
        f"Outer: {outer}, num_x: {num_x}, num_y: {num_y}, num_t: {num_t}\ns0: {s0}, t: {t}, alpha: {alpha}, nu: {nu}, beta: {beta}"
    )

    return outer, num_x, num_y, num_t, s0, t, alpha, nu, beta


def getPrefedinedInputDataSet(size: str):
    """
    Returns one of the predefined (original benchmark) datasets (small/medium/large)

    :param size: It should be 'XS', 'S', 'M' or 'L'
    :return: outer, num_x, num_y, num_t, s0, t, alpha, nu, beta
    """

    assert size.upper() in {'XS', 'S', 'M', 'L'}
    outer = constants.inputs[size.upper()]["Outer"]
    num_x = constants.inputs[size.upper()]["Num_X"]
    num_y = constants.inputs[size.upper()]["Num_Y"]
    num_t = constants.inputs[size.upper()]["Num_T"]
    s0 = constants.inputs[size.upper()]["s0"]
    t = constants.inputs[size.upper()]["t"]
    alpha = constants.inputs[size.upper()]["alpha"]
    nu = constants.inputs[size.upper()]["nu"]
    beta = constants.inputs[size.upper()]["beta"]
    print(
        f"Outer: {outer}, num_x: {num_x}, num_y: {num_y}, num_t: {num_t}\ns0: {s0}, t: {t}, alpha: {alpha}, nu: {nu}, beta: {beta}"
    )

    return outer, num_x, num_y, num_t, s0, t, alpha, nu, beta


def getPrefefinedOutputDataSet(size: str):
    """
    Returns golden result
    """

    assert size.upper() in {'XS', 'S', 'M', 'L'}
    return constants.outputs[size.upper()]
