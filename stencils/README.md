# Stencil Codes

## Climate/Weather

Includes the following two basic kernels for the dynamical core of a climate/weather model:
- hdiff: Horizontal diffusion
- vadv: Vertical advection

## CFDs

Includes the following two Navier-Stokes solvers:
- cavity_flow: 2D cavity flow
- channel_flow: 2D channel flow

## Characteristics

- Memory-bound
- Tests capablity of compiler/backend to optimize memory accesses for different architectures:
  - Loop interchange to move parallelizable loops to the top-level.
  - Change of strides to match the optimal access order of the target architecture.
  - Compression of intermediate arrays to reduce the number of memory accesses.

