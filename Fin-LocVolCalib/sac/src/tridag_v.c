#define VEC 8

typedef float float_vec __attribute__ ((__vector_size__ (32), aligned(4)));

void tridag_v(float_vec a[NUM_X], float_vec b[NUM_X], float_vec c[NUM_X], float_vec y[NUM_X])
{
  int n = NUM_X;
//  float_vec (*a)[NUM_X] = (float_vec (*)[NUM_X])ap;
//  float_vec (*b)[NUM_X] = (float_vec (*)[NUM_X])bp;
//  float_vec (*c)[NUM_X] = (float_vec (*)[NUM_X])cp;
//  float_vec (*y)[NUM_X] = (float_vec (*)[NUM_X])yp;

  /* This is the modified Thomas method from Numerical Methods.
   * Note that the non-zeroes in a row are a, b, c in this application,
   * and b, a, c in Numerical Methods.
   * We store gamma in b. */
  b[0] = 1.0 / b[0];
  y[0] = b[0] * y[0];

  b[1] = 1.0 / b[1];
  y[1] = b[1] * (y[1] - a[1] * y[0]);
  for (int i = 2; i < n; i++) {
      b[i] = 1.0 / (b[i] - a[i] * b[i - 1] * c[i - 1]);
      y[i] = b[i] * (y[i] - a[i] * y[i - 1]);
  }

  for (int i = n - 2; i >= 1; i--) {
      y[i] = y[i] - b[i] * c[i] * y[i + 1];
  }
}
