/**
 * avx2 implementation of matmul satisfying the following assumptions:
 *
 * 1. dimensions are powers of two
 * 2. dimensions are divisible by the vector width
 *
 * Optimised for the following assumptions:
 *
 * 1. The matrices fit in L2.
 * 2. Target is zen2 or zen3
 *
 * Note that zen2, zen3 have 32 KiB L1D and 512 KiB L2
 * Two fma instructions on ymm registers per cycle,
 * L1 can read in two ymm registers and write one per cycle
 * L2 can read in two ymm registers and write one every two cycle
 * That means reusing each element 4 times suffices to bridge the
 * L2 -> register bandwidth.
 *
 * Similar to kernel.c, but C is less square. This is better for the
 * microkernel as we need less broadcasts, but the reuse is less,
 * so this may give problems in a full implementation.
 *
 * For zen2, 4.0 GHz: 114 Gflops/s on bin/matmul 256 128 128 100000
 *                    114 Gflops/s on bin/matmul 256 128 256 100000
 *
 * Peak performance is 128 Gflops/s, so this is about 90%
 * TODO: I don't see why this cannot be 98% as well. Probably messed something
 * up in the streaming from L1. Can prefetch handle the strided access?
 **/

#include <immintrin.h>
#include <cblas.h>
#include <sys/time.h>

#define MR 6
#define NR 16
#define KC 128

float maxf(float a, float b)
{
    return (a > b) ? a : b;
}

/* Computes c = c + ab for a of size [MR, k] and b of size [k, NR].
 * a is a submatrix of a larger matrix with ld_a columns.
 * b, c are submatrices of a larger matrix with ld_c columns. */
inline
void kernel(float *a, float *b, float *c, int k,
            int ld_a, int ld_c)
{
    __m256 c00  = _mm256_loadu_ps(&c[0 * ld_c + 0]);
    __m256 c01  = _mm256_loadu_ps(&c[0 * ld_c + 8]);
    __m256 c10  = _mm256_loadu_ps(&c[1 * ld_c + 0]);
    __m256 c11  = _mm256_loadu_ps(&c[1 * ld_c + 8]);
    __m256 c20  = _mm256_loadu_ps(&c[2 * ld_c + 0]);
    __m256 c21  = _mm256_loadu_ps(&c[2 * ld_c + 8]);
    __m256 c30  = _mm256_loadu_ps(&c[3 * ld_c + 0]);
    __m256 c31  = _mm256_loadu_ps(&c[3 * ld_c + 8]);
    __m256 c40  = _mm256_loadu_ps(&c[4 * ld_c + 0]);
    __m256 c41  = _mm256_loadu_ps(&c[4 * ld_c + 8]);
    __m256 c50  = _mm256_loadu_ps(&c[5 * ld_c + 0]);
    __m256 c51  = _mm256_loadu_ps(&c[5 * ld_c + 8]);

    __m256 ai;  /* a[i, p]   */
    __m256 bp1; /* b[p, 0:7] */
    __m256 bp2; /* b[p, 8:15] */
    for (int p = 0; p < k; p++) {
        bp1 = _mm256_loadu_ps(&b[p * ld_c]);
        bp2 = _mm256_loadu_ps(&b[p * ld_c + 8]);

        ai  = _mm256_broadcast_ss(&a[0 * ld_a + p]);
        c00 = _mm256_fmadd_ps(ai, bp1, c00);
        c01 = _mm256_fmadd_ps(ai, bp2, c01);

        ai  = _mm256_broadcast_ss(&a[1 * ld_a + p]);
        c10 = _mm256_fmadd_ps(ai, bp1, c10);
        c11 = _mm256_fmadd_ps(ai, bp2, c11);

        ai  = _mm256_broadcast_ss(&a[2 * ld_a + p]);
        c20 = _mm256_fmadd_ps(ai, bp1, c20);
        c21 = _mm256_fmadd_ps(ai, bp2, c21);

        ai  = _mm256_broadcast_ss(&a[3 * ld_a + p]);
        c30 = _mm256_fmadd_ps(ai, bp1, c30);
        c31 = _mm256_fmadd_ps(ai, bp2, c31);

        ai  = _mm256_broadcast_ss(&a[4 * ld_a + p]);
        c40 = _mm256_fmadd_ps(ai, bp1, c40);
        c41 = _mm256_fmadd_ps(ai, bp2, c41);

        ai  = _mm256_broadcast_ss(&a[5 * ld_a + p]);
        c50 = _mm256_fmadd_ps(ai, bp1, c50);
        c51 = _mm256_fmadd_ps(ai, bp2, c51);
    }

    _mm256_storeu_ps(&c[0 * ld_c + 0], c00);
    _mm256_storeu_ps(&c[0 * ld_c + 8], c01);
    _mm256_storeu_ps(&c[1 * ld_c + 0], c10);
    _mm256_storeu_ps(&c[1 * ld_c + 8], c11);
    _mm256_storeu_ps(&c[2 * ld_c + 0], c20);
    _mm256_storeu_ps(&c[2 * ld_c + 8], c21);
    _mm256_storeu_ps(&c[3 * ld_c + 0], c30);
    _mm256_storeu_ps(&c[3 * ld_c + 8], c31);
    _mm256_storeu_ps(&c[4 * ld_c + 0], c40);
    _mm256_storeu_ps(&c[4 * ld_c + 8], c41);
    _mm256_storeu_ps(&c[5 * ld_c + 0], c50);
    _mm256_storeu_ps(&c[5 * ld_c + 8], c51);
}

/* Computes c = c + ab for a of size [4, k] and b of size [k, NR].
 * a is a submatrix of a larger matrix with ld_a columns.
 * b, c are submatrices of a larger matrix with ld_c columns. */
inline
void kernel4(float *a, float *b, float *c, int k,
            int ld_a, int ld_c)
{
    __m256 c00  = _mm256_loadu_ps(&c[0 * ld_c + 0]);
    __m256 c01  = _mm256_loadu_ps(&c[0 * ld_c + 8]);
    __m256 c10  = _mm256_loadu_ps(&c[1 * ld_c + 0]);
    __m256 c11  = _mm256_loadu_ps(&c[1 * ld_c + 8]);
    __m256 c20  = _mm256_loadu_ps(&c[2 * ld_c + 0]);
    __m256 c21  = _mm256_loadu_ps(&c[2 * ld_c + 8]);
    __m256 c30  = _mm256_loadu_ps(&c[3 * ld_c + 0]);
    __m256 c31  = _mm256_loadu_ps(&c[3 * ld_c + 8]);

    __m256 ai;  /* a[i, p]   */
    __m256 bp1; /* b[p, 0:7] */
    __m256 bp2; /* b[p, 8:15] */
    for (int p = 0; p < k; p++) {
        bp1 = _mm256_loadu_ps(&b[p * ld_c]);
        bp2 = _mm256_loadu_ps(&b[p * ld_c + 8]);

        ai  = _mm256_broadcast_ss(&a[0 * ld_a + p]);
        c00 = _mm256_fmadd_ps(ai, bp1, c00);
        c01 = _mm256_fmadd_ps(ai, bp2, c01);

        ai  = _mm256_broadcast_ss(&a[1 * ld_a + p]);
        c10 = _mm256_fmadd_ps(ai, bp1, c10);
        c11 = _mm256_fmadd_ps(ai, bp2, c11);

        ai  = _mm256_broadcast_ss(&a[2 * ld_a + p]);
        c20 = _mm256_fmadd_ps(ai, bp1, c20);
        c21 = _mm256_fmadd_ps(ai, bp2, c21);

        ai  = _mm256_broadcast_ss(&a[3 * ld_a + p]);
        c30 = _mm256_fmadd_ps(ai, bp1, c30);
        c31 = _mm256_fmadd_ps(ai, bp2, c31);
    }

    _mm256_storeu_ps(&c[0 * ld_c + 0], c00);
    _mm256_storeu_ps(&c[0 * ld_c + 8], c01);
    _mm256_storeu_ps(&c[1 * ld_c + 0], c10);
    _mm256_storeu_ps(&c[1 * ld_c + 8], c11);
    _mm256_storeu_ps(&c[2 * ld_c + 0], c20);
    _mm256_storeu_ps(&c[2 * ld_c + 8], c21);
    _mm256_storeu_ps(&c[3 * ld_c + 0], c30);
    _mm256_storeu_ps(&c[3 * ld_c + 8], c31);
}

/* Computes c = c + ab for a of size [2, k] and b of size [k, NR].
 * a is a submatrix of a larger matrix with ld_a columns.
 * b, c are submatrices of a larger matrix with ld_c columns. */
inline
void kernel2(float *a, float *b, float *c, int k,
             int ld_a, int ld_c)
{
    __m256 c00  = _mm256_loadu_ps(&c[0 * ld_c + 0]);
    __m256 c01  = _mm256_loadu_ps(&c[0 * ld_c + 8]);
    __m256 c10  = _mm256_loadu_ps(&c[1 * ld_c + 0]);
    __m256 c11  = _mm256_loadu_ps(&c[1 * ld_c + 8]);

    __m256 ai;  /* a[i, p]   */
    __m256 bp1; /* b[p, 0:7] */
    __m256 bp2; /* b[p, 8:15] */
    for (int p = 0; p < k; p++) {
        bp1 = _mm256_loadu_ps(&b[p * ld_c]);
        bp2 = _mm256_loadu_ps(&b[p * ld_c + 8]);

        ai  = _mm256_broadcast_ss(&a[0 * ld_a + p]);
        c00 = _mm256_fmadd_ps(ai, bp1, c00);
        c01 = _mm256_fmadd_ps(ai, bp2, c01);

        ai  = _mm256_broadcast_ss(&a[1 * ld_a + p]);
        c10 = _mm256_fmadd_ps(ai, bp1, c10);
        c11 = _mm256_fmadd_ps(ai, bp2, c11);
    }

    _mm256_storeu_ps(&c[0 * ld_c + 0], c00);
    _mm256_storeu_ps(&c[0 * ld_c + 8], c01);
    _mm256_storeu_ps(&c[1 * ld_c + 0], c10);
    _mm256_storeu_ps(&c[1 * ld_c + 8], c11);
}

void matmul(float *a, float *b, float *c, int m, int k, int n)
{
    int i = 0;
    for (; i + MR < m; i += MR) {
        for (int p = 0; p < k; p += KC) {
            for (int j = 0; j < n; j += NR) {
                kernel(a + i * k + p, b + p * n + j, c + i * n + j, KC, k, n);
            }
        }
    }

    /* 2^n mod 6 is either 2 or 4. */
    if (m - i == 4) {
        for (int p = 0; p < k; p += KC) {
            for (int j = 0; j < n; j += NR) {
                kernel4(a + i * k + p, b + p * n + j, c + i * n + j, KC, k, n);
            }
        }
    } else if (m - i == 2) {
        for (int p = 0; p < k; p += KC) {
            for (int j = 0; j < n; j += NR) {
                kernel2(a + i * k + p, b + p * n + j, c + i * n + j, KC, k, n);
            }
        }
    }  else {
        printf("Tell thomas to brush up on his modulo arithmetic!\n");
        abort();
    }
}

float test_matmul(int m, int k, int n)
{
    if (m % 2 != 0 || k % KC != 0 || n % NR != 0) {
        printf("This implementation assumes the 2 | m, %d | k, %d | n\n",
                KC, NR);
        abort();
    }

    float *a = malloc(m * k * sizeof(float));
    float *b = malloc(k * n * sizeof(float));
    float *c = malloc(m * n * sizeof(float));
    float *c_check = malloc(m * n * sizeof(float));

    srand(time(NULL));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            a[i * k + j] = (float)rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            b[i * n + j] = (float)rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float x = (float)rand() / RAND_MAX;
            c[i * n + j] = c_check[i * n + j] = x;
        }
    }

    matmul(a, b, c, m, k, n);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0, a, k, b, n, 1.0, c_check, n);

    float max_diff = 0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float diff = fabs(c[i * n + j] - c_check[i * n + j]);
            max_diff = maxf(max_diff, diff);
        }
    }

    free(a);
    free(b);
    free(c);
    free(c_check);

    return max_diff;
}

double bench_matmul(int m, int k, int n, int iter)
{
    float *a = calloc(m * k, sizeof(float));
    float *b = calloc(k * n, sizeof(float));
    float *c = calloc(m * n, sizeof(float));

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

    for (int t = 0; t < iter; t++) {
        matmul(a, b, c, m, k, n);
    }

    gettimeofday(&tv2, NULL);
    double duration = (double)(tv2.tv_usec - tv1.tv_usec) / 1e6 +
                      (double)(tv2.tv_sec - tv1.tv_sec);

    free(a);
    free(b);
    free(c);

    return 2.0 * m * k * n * iter / duration / 1e9;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        printf("Takes four argument: m, k, n, number of times to run kernel\n");
        return EXIT_FAILURE;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    int iter = atoi(argv[4]);

    printf("Error for k = %d is %f\n", k, test_matmul(m, k, n));
    printf("Compute rate %lf Gflops/s\n", bench_matmul(m, k, n, iter));

    return EXIT_SUCCESS;
}
