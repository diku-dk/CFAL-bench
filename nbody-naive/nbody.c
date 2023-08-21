#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <time.h>

long now() {
  struct timespec tv;
  assert(clock_gettime(CLOCK_MONOTONIC, &tv) == 0);
  return tv.tv_sec * 1e9L + tv.tv_nsec;
}

#define EPSILON 1e-9f

struct particle {
  double x, y, z;
  double vx, vy, vz;
  double m;
};

void nbody_step(int n, double dt, struct particle* ps) {
  // First update velocities.
  for (int i = 0; i < n; i++) {
    double fx = 0.0f; double fy = 0.0f; double fz = 0.0f;

    for (int j = 0; j < n; j++) {
      double dx = ps[j].x - ps[i].x;
      double dy = ps[j].y - ps[i].y;
      double dz = ps[j].z - ps[i].z;
      double dist_sqr = dx*dx + dy*dy + dz*dz + EPSILON;
      double inv_dist = 1.0f / sqrtf(dist_sqr);
      double inv_dist3 = inv_dist * inv_dist * inv_dist;

      fx += dx * inv_dist3;
      fy += dy * inv_dist3;
      fz += dz * inv_dist3;
    }

    ps[i].vx += dt*fx;
    ps[i].vy += dt*fy;
    ps[i].vz += dt*fz;
  }
  // Then adjust positions.
  for (int i = 0 ; i < n; i++) {
    ps[i].x += ps[i].vx*dt;
    ps[i].y += ps[i].vy*dt;
    ps[i].z += ps[i].vz*dt;
  }
}

struct particle* read_particles(int* n_out) {
  int n = 0, capacity = 1000;
  struct particle *ps = calloc(capacity, sizeof(struct particle));

  struct particle p = { .vx = 0, .vy = 0, .vz = 0 };
  while (scanf("%lf %lf %lf %lf", &p.x, &p.y, &p.z, &p.m) == 4) {
    if (n == capacity) {
      capacity *= 2;
      ps = realloc(ps, capacity * sizeof(struct particle));
    }
    ps[n++] = p;
  }

  *n_out = n;
  return ps;
}

void print_particles(int n, struct particle* ps) {
  for (int i = 0; i < n; i++) {
    printf("%f %f %f\n", ps[i].x, ps[i].y, ps[i].z);
  }
}

int main() {
  int n, k;
  double dt;
  scanf("%d %lf", &k, &dt);

  struct particle* ps = read_particles(&n);

  fprintf(stderr, "Iterations: %d\nTimestep: %f\nParticles: %d\n", k, dt, n);

  long before = now();
  for (int i = 0; i < k; i++) {
    nbody_step(n, dt, ps);
  }
  long after = now();
  fprintf(stderr, "Microseconds: %f\n", (after-before)/1000.0);
  print_particles(n, ps);
}
