"""

    Contains constants and pre-defined input data set (as defined in the Original Benchmark)

"""

WITH_FLOATS = 1
REAL3_CT = 4
TRANSPOSE_UV = 1

BLOCK_DIM = 16
LOGWORKGROUP_SIZE = 8
WORKGROUP_SIZE = 1 << LOGWORKGROUP_SIZE

inputs = {
    "S": {
        "Outer": 16,
        "Num_X": 32,
        "Num_Y": 256,
        "Num_T": 256,
        "s0": 0.03,
        "t": 5.0,
        "alpha": 0.2,
        "nu": 0.6,
        "beta": 0.5
    },
    "M": {
        "Outer": 128,
        "Num_X": 256,
        "Num_Y": 32,
        "Num_T": 64,
        "s0": 0.03,
        "t": 5.0,
        "alpha": 0.2,
        "nu": 0.6,
        "beta": 0.5
    },
    "L": {
        "Outer": 256,
        "Num_X": 256,
        "Num_Y": 256,
        "Num_T": 64,
        "s0": 0.03,
        "t": 5.0,
        "alpha": 0.2,
        "nu": 0.6,
        "beta": 0.5
    }
}
