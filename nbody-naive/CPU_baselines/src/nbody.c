#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <immintrin.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <immintrin.h>

#define EPSILON2 0x1p-53

#define min(a, b) ((a) < (b) ? a : b)

typedef struct {
    double x;
    double y;
    double z;
} Point;

typedef struct {
    double *x;
    double *y;
    double *z;
} Points;

Points alloc_points(int n)
{
    Points p;
    p.x = aligned_alloc(64, n * sizeof(double));
    p.y = aligned_alloc(64, n * sizeof(double));
    p.z = aligned_alloc(64, n * sizeof(double));
    return p;
}

void free_points(Points p)
{
    free(p.x);
    free(p.y);
    free(p.z);
}

void init(Points positions, Points velocities, double *masses, int n)
{
    for (int i = 0; i < n; i++) {
        positions.x[i] = i;
        positions.y[i] = 2.0 * i;
        positions.z[i] = 3.0 * i;
        velocities.x[i] = 0.0;
        velocities.y[i] = 0.0;
        velocities.z[i] = 0.0;
        masses[i] = 1.0;
    }
}

double pow3(double x)
{
    return x * x * x;
}

/* Advances the n-bodies in place by iterations.
 * 19n^2 + 12n flops */
void advance(Points positions, Points velocities, double *masses,
             double dt, int n, int iterations)
{
    #pragma omp parallel
    {
        for (int t = 0; t < iterations; t++) {
            /* omp for has implicit barrier at the end */
            #pragma omp for
            for (int i = 0; i < n; i++) {
                double ax = 0.0;
                double ay = 0.0;
                double az = 0.0;
                /* Loop body is 19 flops (assuming masses[j] / norm is computed
                 * only once) */
                for (int j = 0; j < n; j++) {
                    double bufx = positions.x[j] - positions.x[i];
                    double bufy = positions.y[j] - positions.y[i];
                    double bufz = positions.z[j] - positions.z[i];
                    /* norm = ||positions[i] - positions[j]||^3 */
                    double norm = pow3(sqrt(bufx * bufx +
                                            bufy * bufy +
                                            bufz * bufz) + EPSILON2);
                    ax += bufx * masses[j] / norm;
                    ay += bufy * masses[j] / norm;
                    az += bufz * masses[j] / norm;
                }
                velocities.x[i] += ax * dt;
                velocities.y[i] += ay * dt;
                velocities.z[i] += az * dt;
            }

            #pragma omp for
            for (int i = 0; i < n; i++) {
                positions.x[i] += velocities.x[i] * dt;
                positions.y[i] += velocities.y[i] * dt;
                positions.z[i] += velocities.z[i] * dt;
            }
        }
    }
}

double sum_points(Points positions, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += positions.x[i];
        sum += positions.y[i];
        sum += positions.z[i];
    }
    return sum;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: n, iterations\n");
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    int iterations = atoi(argv[2]);

    Points positions = alloc_points(n);
    Points velocities = alloc_points(n);
    double *masses = (double *)malloc(n * sizeof(double));

    init(positions, velocities, masses, n);

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    advance(positions, velocities, masses, 0.01, n, iterations);
    gettimeofday(&tv2, NULL);
    double duration = (double) (tv2.tv_usec - tv1.tv_usec) / 1e6 +
                      (double) (tv2.tv_sec - tv1.tv_sec);

#ifdef DEBUG
    fprintf(stderr, "Bodies\n");
    for (int i = 0; i < n; i++) {
        fprintf(stderr, "(%.17e, %.17e, %.17e)\n", 
                    positions.x[i], positions.y[i], positions.z[i]);
    }
#endif

    fprintf(stderr, "Nbody with %d bodies, %d iterations.\n"
                    "This took %lfs.\n"
                    "Compute rate in Gflops/s: ",
                    n, iterations, duration);
    fflush(stderr);
    printf("%lf\n", (19.0 * n * n + 12.0 * n) * iterations / 1e9 / duration);

    free_points(positions);
    free_points(velocities);
    free(masses);

    return EXIT_SUCCESS;
}
