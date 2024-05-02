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
    double *x;
    double *y;
    double *z;
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

/* accel[i] = sum_j m[j] (pos[i] - pos[j]) / ||pos[i] - pos[j]||^3 */
/* 18 n^2 flops */
void accelerateAll(Points accel, Points positions, double *masses, int n)
{
    for (int i = 0; i < n; i++) {
        double ax = 0.0;
        double ay = 0.0;
        double az = 0.0;
        /* Loop body is (worst case n != 0) 18 flops */
        for (int j = 0; j < n; j++) {
            double bufx = positions.x[j] - positions.x[i];
            double bufy = positions.y[j] - positions.y[i];
            double bufz = positions.z[j] - positions.z[i];
            /* norm = ||positions[i] - positions[j]||^3 */
            double norm = pow3(sqrt(bufx * bufx +
                                    bufy * bufy +
                                    bufz * bufz));
            if (norm != 0.0) {
                ax += bufx * masses[j] / norm;
                ay += bufy * masses[j] / norm;
                az += bufz * masses[j] / norm;
            }
        }
        accel.x[i] = ax;
        accel.y[i] = ay;
        accel.z[i] = az;
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
    Points accel = alloc_points(n);
    double *masses = (double *)malloc(n * sizeof(double));

    init(positions, velocities, masses, n);

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    for (int i = 0; i < iterations; i++) {
        advance(positions, velocities, masses, accel, 0.01, n);
    }
    gettimeofday(&tv2, NULL);
    double duration = (double) (tv2.tv_usec - tv1.tv_usec) / 1e6 +
                      (double) (tv2.tv_sec - tv1.tv_sec);

    fprintf(stderr, "Bodies\n");
    for (int i = 0; i < n; i++) {
        fprintf(stderr, "(%.17e, %.17e, %.17e)\n", 
                    positions.x[i], positions.y[i], positions.z[i]);
    }

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
