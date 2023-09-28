"""
Simple 1:1 porting of C sequential nbody-naive (sequential/nbody.c).
"""

import numpy as np
import math
import time

import argparse
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


def dump_input_particles(
    N,
    dt,
    pos,
    mass,
    filename: str = "dump.txt",
):
    """
    Dump input particles to file, using the same format used by the C sequential impl

    """
    with open(filename, "w") as file:
        file.write(f"{N} {dt}\n")
        for i in range(len(pos)):
            file.write(f"{pos[i][0]} {pos[i][1]} {pos[i][2]} {mass[i][0]}\n")
    file.close()

    pass


def print_particles(positions):
    """
    Prints particle positions
    """
    for p in positions:
        print(f"{p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}")


def nbody_step(dt: np.float64, pos, vel, mass):
    """
    Computes a step of nbody

    """
    Np = len(pos)
    # Update velocities
    for i in range(Np):
        fx = 0.0
        fy = 0.0
        fz = 0.0

        for j in range(Np):
            dx = pos[j][0] - pos[i][0]
            dy = pos[j][1] - pos[i][1]
            dz = pos[j][2] - pos[i][2]
            dist_sqr = dx**2 + dy**2 + dz**2 + 1e-9
            inv_dist = 1.0 / math.sqrt(dist_sqr)
            inv_dist3 = inv_dist**3

            # accelerations
            fx += dx * mass[j] * inv_dist3
            fy += dy * mass[j] * inv_dist3
            fz += dz * mass[j] * inv_dist3

        vel[i][0] += dt * fx
        vel[i][1] += dt * fy
        vel[i][2] += dt * fz

    # adjust positions
    for i in range(Np):
        pos[i][0] += vel[i][0] * dt
        pos[i][1] += vel[i][1] * dt
        pos[i][2] += vel[i][2] * dt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('number_iterations', type=int)
    parser.add_argument('number_particles', type=int)
    parser.add_argument('dt', type=float)
    args = vars(parser.parse_args())
    N = args["number_iterations"]  # number of iteration
    Nparticles = args["number_particles"]
    dt = args["dt"]

    mass, pos, vel = init_data(Nparticles)
    dump_input_particles(N, dt, pos, mass)

    # Execute nbody
    start = time.time()

    for i in range(N):
        nbody_step(dt, pos, vel, mass)
    end = time.time()

    print(f"Time in usecs: {(end-start)*1e6}")

    print_particles(pos)