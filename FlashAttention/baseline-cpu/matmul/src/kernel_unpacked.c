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
 * Similar to kernel_optimised_zen2.c, but a, b, and c can be submatrices.
 * As a, b, c are already assumed to be in L2, packing may not be worthwhile.
 *
 * Same performance
 **/

#include <immintrin.h>
#include <cblas.h>
#include <sys/time.h>

#define MR 6
#define NR 16

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

float test_kernel(int kc)
{
    int m = 10 * MR;
    int k = 10 * kc;
    int n = 10 * NR;
    float *a = malloc(m * k * sizeof(float));
    float *b = malloc(k * n * sizeof(float));
    float *c = malloc(m * n * sizeof(float));
    float *c_check = malloc(m * n * sizeof(float));

    srand(time(NULL));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
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

    float *sub_c = c + 4;
    float *sub_c_check = c_check + 4;
    kernel(a, b, sub_c, kc, k, n);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                MR, NR, kc, 1.0, a, k, b, n, 1.0, sub_c_check, n);

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

double bench_kernel(int kc, int iter)
{
    int m = 10 * MR;
    int k = 10 * kc;
    int n = 10 * NR;
    float *a = calloc(m * k, sizeof(float));
    float *b = calloc(k * n, sizeof(float));
    float *c = calloc(m * n, sizeof(float));

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

    for (int t = 0; t < iter; t++) {
        kernel(a + 8, b + 2, c + 5, kc, k, n);
    }

    gettimeofday(&tv2, NULL);
    double duration = (double)(tv2.tv_usec - tv1.tv_usec) / 1e6 +
                      (double)(tv2.tv_sec - tv1.tv_sec);

    free(a);
    free(b);
    free(c);

    return 2.0 * MR * kc * NR * iter / duration / 1e9;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Takes two argument: shared dimension, "
                "number of times to run kernel\n");
        return EXIT_FAILURE;
    }

    int k = atoi(argv[1]);
    int iter = atoi(argv[2]);
    printf("Error for k = %d is %f\n", k, test_kernel(k));
    printf("Compute rate %lf Gflops/s\n", bench_kernel(k, iter));

    return EXIT_SUCCESS;
}
