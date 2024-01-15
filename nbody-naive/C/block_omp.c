#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>
#include <sys/time.h>

#define min(a, b) ((a) < (b) ? a : b)

/* BLOCK bodies should comfortably fit inside L1,
 * should also be a multiple of the vector width. 
 * For n = 10000, iter = 10 we get
 *
 * BLOCK    Gflops/s
 * 1000     19,2
 * 2000     21,6
 * 2500     23,0
 * 3000     17,3
 *
 * This is a little surprising as 2500 gives more
 * L1 cache misses than 2000.
 */
#define BLOCK 2500

typedef struct {
    double x;
    double y;
    double z;
} Point;

typedef struct {
    double * __restrict__ x;
    double * __restrict__ y;
    double * __restrict__ z;
} Points;

Points alloc_points(int n)
{
    Points p;
    p.x = malloc(n * sizeof(double));
    p.y = malloc(n * sizeof(double));
    p.z = malloc(n * sizeof(double));
    return p;
}

void free_points(Points p)
{
    free(p.x);
    free(p.y);
    free(p.z);
}

void init(Points positions, int n)
{
    for (int i = 0; i < n; i++) {
        positions.x[i] = i;
        positions.y[i] = i;
        positions.z[i] = i;
    }
}

int roundDown(int a, int b)
{
    return a / b * b;
}

double pow3(double x)
{
    return x * x * x;
}

/* Computes accel[start : end - 1]. */
void accelerateBodies(Points accel, Points positions, double *masses, 
                      int start, int end, int n)
{
    for (int i = start; i < end; i++) {
        accel.x[i] = 0.0;
        accel.y[i] = 0.0;
        accel.z[i] = 0.0;
    }

    for (int I = start; I < end; I += BLOCK) {
        /* To avoid the j < n check in the tight inner loop.
         * The remainder of the sum is computed in a new loop. 
         * Also makes sure the compiler does not need to worry about
         * the case where n does not divide the vector width. */
        for (int J = 0; J + BLOCK < n; J += BLOCK) {
    for (int i = I; i < I + BLOCK && i < end; i++) {
        /* Loop body is (worst case n != 0) 18 flops */
        double ax = 0;
        double ay = 0;
        double az = 0;
        for (int j = J; j < J + BLOCK; j++) {
            Point buf;
            buf.x = positions.x[i] - positions.x[j];
            buf.y = positions.y[i] - positions.y[j],
            buf.z = positions.z[i] - positions.z[j];
            /* n = ||positions[i] - positions[j]||^3 */
            double n = pow3(sqrt(buf.x * buf.x +
                                 buf.y * buf.y +
                                 buf.z * buf.z));
            if (n == 0) {
                buf.x = 0.0;
                buf.y = 0.0;
                buf.z = 0.0;
            } else {
                buf.x *= masses[j] / n;
                buf.y *= masses[j] / n;
                buf.z *= masses[j] / n;
            }
            ax += buf.x;
            ay += buf.y;
            az += buf.z;
        }
        accel.x[i] += ax;
        accel.y[i] += ay;
        accel.z[i] += az;
    }
        }
    }

    for (int i = start; i < end; i++) {
        double ax = 0;
        double ay = 0;
        double az = 0;
        for (int j = roundDown(n, BLOCK); j < n; j++) {
            Point buf;
            buf.x = positions.x[i] - positions.x[j];
            buf.y = positions.y[i] - positions.y[j],
            buf.z = positions.z[i] - positions.z[j];
            /* n = ||positions[i] - positions[j]||^3 */
            double n = pow3(sqrt(buf.x * buf.x +
                                 buf.y * buf.y +
                                 buf.z * buf.z));
            if (n == 0) {
                buf.x = 0.0;
                buf.y = 0.0;
                buf.z = 0.0;
            } else {
                buf.x *= masses[j] / n;
                buf.y *= masses[j] / n;
                buf.z *= masses[j] / n;
            }
            ax += buf.x;
            ay += buf.y;
            az += buf.z;
        }
        accel.x[i] += ax;
        accel.y[i] += ay;
        accel.z[i] += az;       
    }
}

/* Advances the n-bodies in place. accel is a buffer, does not need to be
 * initialized.
 * 18n^2 + 12n flops */
void advance(Points positions, Points velocities, double *masses,
             Points accel, double dt, int n)
{
    #pragma omp parallel
    {
        int p = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int omp_block_size = (n + p - 1) / p;
        int start = tid * omp_block_size;
        int end = min((tid + 1) * omp_block_size, n);
        accelerateBodies(accel, positions, masses, start, end, n);
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        velocities.x[i] += accel.x[i] * dt;
        velocities.y[i] += accel.y[i] * dt;
        velocities.z[i] += accel.z[i] * dt;
        positions.x[i] += velocities.x[i] * dt;
        positions.y[i] += velocities.y[i] * dt;
        positions.z[i] += velocities.z[i] * dt;
    }
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
    Points accel = alloc_points(n);
    double *masses = (double *)malloc(n * sizeof(double));

    init(positions, n);

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    for (int i = 0; i < iterations; i++) {
        advance(positions, velocities, masses, accel, 0.1, n);
    }
    gettimeofday(&tv2, NULL);
    double duration = (double) (tv2.tv_usec - tv1.tv_usec) / 1e6 +
                      (double) (tv2.tv_sec - tv1.tv_sec);

    fprintf(stderr, "Sequential nbody with %d bodies, %d iterations.\n"
                    "This took %lfs.\n"
                    "Compute rate in Gflops/s: ",
                    n, iterations, duration);
    printf("%lf\n", (18.0 * n * n + 12.0 * n) * iterations / 1e9 / duration);

    free_points(positions);
    free_points(velocities);
    free_points(accel);
    free(masses);

    return EXIT_SUCCESS;
}
