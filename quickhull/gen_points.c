// Usage:
//
// $ ./gen_points n m k s > file.dat
//
// Generates an n x m board with k random, distinct points.
//
// s is either c (circle), r (rectangular) or q (quadratic curve),
// denoting in which shape the points must be taken.
//
// Ported from code by Ivo Gabe de Wolff.
//
// This program does not guarantee the absence of duplicate points,
// but it is very unlikely given the size of the space.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>

void usage(const char* argv0) {
  fprintf(stderr, "Usage: %s N M K <c|r|q>\n", argv0);
  exit(1);
}

double square(double x) {
  return x*x;
}

int randint(int upper) {
  return rand() % (upper+1);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    usage(argv[0]);
  }

  int n = atoi(argv[1]);
  int m = atoi(argv[2]);
  int k = atoi(argv[3]);
  bool circle = false;
  bool quadratic = false;

  switch (argv[4][0]) {
  case 'c':
    circle = true;
    break;
  case 'q':
    quadratic = true;
    break;
  case 'r':
    break;
  default:
    usage(argv[0]);
  }

  assert(n <= RAND_MAX);
  assert(m <= RAND_MAX);

  srand(123);

  for (int i = 0; i < k; i++) {
    int x, y;

    x = randint(n);
    if (quadratic) {
      y = square(x/(double)n)*m;
    } else {
      y = randint(m);
    }
    if (circle) {
      while (square(x/(double)n - 0.5) + square(y/(double)m - 0.5) > square(0.5)) {
        x = randint(n);
        y = randint(m);
      }
    }
    fwrite(&x, sizeof(int), 1, stdout);
    fwrite(&y, sizeof(int), 1, stdout);
  }
}
