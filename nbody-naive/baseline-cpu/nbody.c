#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

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
/* 19 n^2 flops */
void accelerateAll(Points accel, Points positions, double *masses, int n)
{
#pragma omp parallel for
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
    accel.x[i] = ax;
    accel.y[i] = ay;
    accel.z[i] = az;
  }
}

/* Advances the n-bodies in place. accel is a buffer, does not need to be
 * initialized.
 * 19n^2 + 12n flops */
void advance(Points positions, Points velocities, double *masses,
             Points accel, double dt, int n)
{
  accelerateAll(accel, positions, masses, n);

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
  if (argc != 4) {
    printf("Usage: N STEPS RUNS\n");
    return EXIT_FAILURE;
  }

  int n = atoi(argv[1]);
  int steps = atoi(argv[2]);
  int runs = atoi(argv[3]);

  fprintf(stderr,
          "OpenMP nbody with %d bodies, %d steps, %d runs.\n",
          n, steps, runs);

  Points positions = alloc_points(n);
  Points velocities = alloc_points(n);
  Points accel = alloc_points(n);
  double *masses = (double *)malloc(n * sizeof(double));

  init(positions, velocities, masses, n);

  double* runtimes = (double*)calloc(runs,sizeof(double));

  for (int r = 0; r < runs; r++) {
    fprintf(stderr, "Run %d\n", r);
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    for (int i = 0; i < steps; i++) {
      advance(positions, velocities, masses, accel, 0.01, n);
    }
    gettimeofday(&tv2, NULL);
    double duration = (double) (tv2.tv_usec - tv1.tv_usec) / 1e6 +
      (double) (tv2.tv_sec - tv1.tv_sec);
    runtimes[r] = duration;
  }

  for (int r = 0; r < runs; r++) {
      double gflops = 1e-9 * (19 * n * n + 12 * n) * steps;
      printf("Baseline (CPU),n=%d,%f\n", n, gflops / (double)runtimes[r]);
  }

  free_points(positions);
  free_points(velocities);
  free_points(accel);
  free(masses);

  return EXIT_SUCCESS;
}
