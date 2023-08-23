##  Turbulent Kinetic Energy (TKE) routine in Veros 

Veros is aframework for ocean modelling:

[1] D. Häfner, R. L. Jacobsen, C. Eden, M. R. B. Kristensen, M. Jochum, R. Nuterman, and B. Vinter, “Veros v0.1 – a fast and versatile ocean simulator in pure python,” Geoscientific Model Development, vol. 11, no. 8, pp. 3299–3312, 2018, [link](https://gmd.copernicus.org/articles/11/3299/2018/)

Rationale:

- TKE is essentially a stencil computation, but less boring because it also involves a tridiagonal solver (TRIDIAG).

- if needed, TRIDIAG can be parallelized by means of scans with 2x2 matrix multiplication and linear function composition.

Please consult [this github repository](https://github.com/Sefrin/ocean_modelling/tree/master) for JAX and Futhark code, and [this report](https://github.com/Sefrin/ocean_modelling/blob/master/report/main.pdf) for high-level rationale. 

