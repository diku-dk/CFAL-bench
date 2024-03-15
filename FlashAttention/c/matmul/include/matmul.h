#ifndef _MATMUL_SDKLFJDF_H_
#define _MATMUL_SDKLFJDF_H_

#define MR 6
#define NR 16
#define KC 64

/* c = c + ab 
 * assumes 2 | m, KC | k, NR | n */
void matmul(float *a, float *b, float *c, int m, int k, int n);

#endif
