# FinPar: Local Volatility Calibration

## High-Level Description and Available Implementations

This is an application benchmark from the computational finance domain --- a simplfied version of code used in production (received from industry). It refers to stochastic volatility calibration, i.e., given a set of (observed) prices of contracts, the task is to identify the parameters of a model of such prices, as a function of volatility (unknown), time and strikes (known), and unobserved parameters like alpha, beta, nu, etc. The volatility is modelled as a system of continuous partial differential equations, which are solved via Crank-Nicolson's finite differences method. The model is a variation of SABR.

- Haskell, C, C+OpenMP and OpenCL implementations are available at [FinPar's github repository](https://github.com/HIPERFIT/finpar/tree/master/LocVolCalib). We propose the version that exploits all parallelism, i.e., `AllParOpenCLMP`

- [A corresponding Futhark implementation is available here](https://github.com/diku-dk/futhark-benchmarks/blob/master/finpar/LocVolCalib.fut)

- The benchmark is presented in Chapter 4 of the paper: ["FinPar: A Parallel Financial Benchmark](https://dl.acm.org/doi/pdf/10.1145/2898354)


## Challenges

There are two key aspects that make this benchmark interesting (from a parallelization perspective):

- it uses the [Tridiagonal-Solver Algorithm](https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm) whose parallelization requires scans with linear-function composition and two-by-two matrix multiplication as operators.

- it has a non-trivial nested-parallel structure (sketched below), that requires flattening in order to exploit all available parallelism (this can be achieved with a combination of map-loop interchange and map fission)

```
map
    loop
        map
          map
        
        map
          tridiag
          
        map
          map
          
        map
          tridiag
```

