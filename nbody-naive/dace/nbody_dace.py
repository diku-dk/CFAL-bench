"""
    DaCe and numpy version of nbody-naive
"""
import numpy as np
import math
import time
import argparse
import dace
from dace.transformation.auto.auto_optimize import auto_optimize
"""
Each particle is represented by its:
- position (3 values, x, y, and z in the original code)
- velocities (3 values, vx, vy, vz in the original code)
- mass (1 value m)
All values are in double precision.

These are organized in three different multi-dimensional array (size Nx3 or Nx1)
"""


def init_data(N, total_mass=20.0):
    """
    Generates N random particles
    """
    from numpy.random import default_rng
    rng = default_rng(42)
    mass = total_mass * np.ones((N, 1)) / N  # total mass is defined
    pos = rng.random((N, 3))  # randomly selected positions and velocities
    # vel = rng.random((N, 3))
    vel = np.zeros((N, 3))  # either zeros o random values
    return mass, pos, vel


N, Np = (dace.symbol(s, dtype=dace.int32) for s in ('N', 'Np'))


def nbody_step_np(dt: np.float64, pos, vel, mass):
    """
    Computes a step of nbody
    (numpy version)
    """

    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: [i,j] value = particle_j - particle_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    dist_sqr = (dx**2 + dy**2 + dz**2 + 1e-9)
    inv_dist3 = dist_sqr**(-1.5)

    fx = (dx * inv_dist3) @ mass
    fy = (dy * inv_dist3) @ mass
    fz = (dz * inv_dist3) @ mass

    a = np.ndarray((pos.shape[0], 3), dtype=np.float64)
    a[:, 0] = fx[:, 0]
    a[:, 1] = fy[:, 0]
    a[:, 2] = fz[:, 0]

    vel += a * dt
    pos += vel * dt

    return pos, vel


def nbody_np(dt, pos, vel, mass):
    for i in range(N):
        pos, vel = nbody_step_np(dt, pos, vel, mass)


@dace.program
def nbody_step(dt: dace.float64, pos: dace.float64[Np, 3], vel: dace.float64[Np, 3], mass: dace.float64[Np]):
    """
    Computes a step of nbody
    """

    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: [i,j] value = particle_j - particle_i
    dx = np.add.outer(-x, x)
    dy = np.add.outer(-y, y)
    dz = np.add.outer(-z, z)

    dist_sqr = (dx**2 + dy**2 + dz**2 + 1e-9)
    inv_dist3 = dist_sqr**(-1.5)

    fx = (dx * inv_dist3) @ mass
    fy = (dy * inv_dist3) @ mass
    fz = (dz * inv_dist3) @ mass

    # pack together the acceleration components
    a = np.ndarray((Np, 3), dtype=np.float64)
    a[:, 0] = fx
    a[:, 1] = fy
    a[:, 2] = fz

    # A seems to be full of zeros because fx, fz is full of zeros
    vel += a * dt
    pos += vel * dt
    return pos, vel


@dace.program
def nbody(dt: dace.float64, pos: dace.float64[Np, 3], vel: dace.float64[Np, 3], mass: dace.float64[Np]):
    for i in range(N):
        pos, vel = nbody_step(dt=dt, pos=pos, vel=vel, mass=mass)
    return pos


def print_particles(positions):
    """
    Prints particle positions
    """
    for p in positions:
        print(f"{p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('number_iterations', type=int, help="Number of iterations")
    parser.add_argument('number_particles', type=int, help="Number of particles")
    parser.add_argument('dt', type=float)
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help="Target platform")
    parser.add_argument("-v", "--validate", default=False, action="store_true", help="Validate result (may be slow)")
    args = vars(parser.parse_args())
    N = args["number_iterations"]
    Np = args["number_particles"]
    dt = args["dt"]
    target = args["target"]
    validate = args["validate"]

    mass, pos, vel = init_data(Np)

    mass_ref = np.copy(mass)
    pos_ref = np.copy(pos)
    vel_ref = np.copy(vel)

    sdfg = nbody.to_sdfg(simplify=True)

    # Apply auto-opt
    # TODO: decide whether to make this optional
    if target == "cpu":
        sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.CPU)
    else:
        sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    sdfg.compile()

    # Execute nbody
    # TODO: decide how to take time
    start = time.time()
    pos = sdfg(dt=dt, pos=pos, vel=vel, mass=mass, N=N, Np=Np)
    end = time.time()

    if validate:
        # check with numpy version
        nbody_np(dt, pos_ref, vel_ref, mass_ref)
        assert np.allclose(pos, pos_ref)
        print("Results validated!")

    print(f"Time in usecs: {(end-start)*1e6}")
