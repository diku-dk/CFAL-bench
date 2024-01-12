#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

#define min(a, b) ((a) < (b) ? a : b)

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

double pow3(double x)
{
    return x * x * x;
}

/* 18 n^2 flops */
void accelerateAll(Points accel, Points positions, double *masses, int n)
{
    for (int i = 0; i < n; i++) {
        accel.x[i] = 0.0;
        accel.y[i] = 0.0;
        accel.z[i] = 0.0;
        /* Loop body is (worst case n != 0) 18 flops */
        for (int j = 0; j < n; j++) {
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
            accel.x[i] += buf.x;
            accel.y[i] += buf.y;
            accel.z[i] += buf.z;
        }
    }
}

/* Advances the n-bodies in place. accel is a buffer, does not need to be 
 * initialized.
 * 18n^2 + 12n flops */
void advance(Points positions, Points velocities, double *masses,
             Points accel, double dt, int n)
{
    accelerateAll(accel, positions, masses, n);

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
