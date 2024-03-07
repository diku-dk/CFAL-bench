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
 * For zen2, 3.7 GHz: 80 Gflops/s for k variable. If statically known,
 * this becomes 90 Gflops/s
 *
 * Peak performance is 118 Gflops/s, so 25% is left on the table.
 **/

#include <immintrin.h>
#include <cblas.h>
#include <sys/time.h>

#define MR 12 
#define NR 8
#define KC 128

float maxf(float a, float b)
{
    return (a > b) ? a : b;
}

/* Computes c = c + ab for a of size [MR, k] and b of size [k, NR].
 * c is a submatrix of a larger matrix with ld_c columns. */
inline
void kernel(float *a, float *b, float *c, int k, int ld_c)
{
    k = KC;

    __m256 c0  = _mm256_loadu_ps(&c[0  * ld_c]);
    __m256 c1  = _mm256_loadu_ps(&c[1  * ld_c]);
    __m256 c2  = _mm256_loadu_ps(&c[2  * ld_c]);
    __m256 c3  = _mm256_loadu_ps(&c[3  * ld_c]);
    __m256 c4  = _mm256_loadu_ps(&c[4  * ld_c]);
    __m256 c5  = _mm256_loadu_ps(&c[5  * ld_c]);
    __m256 c6  = _mm256_loadu_ps(&c[6  * ld_c]);
    __m256 c7  = _mm256_loadu_ps(&c[7  * ld_c]);
    __m256 c8  = _mm256_loadu_ps(&c[8  * ld_c]);
    __m256 c9  = _mm256_loadu_ps(&c[9  * ld_c]);
    __m256 c10 = _mm256_loadu_ps(&c[10 * ld_c]);
    __m256 c11 = _mm256_loadu_ps(&c[11 * ld_c]);

    /* Use two registers for pipelining */
    __m256 ai;  /* a[i, p]   */
    __m256 ai2; /* a[i, p]   */
    __m256 bp;  /* b[p, 0:7] */
    for (int p = 0; p < k; p++) {
        bp = _mm256_loadu_ps(&b[p * NR]);

        ai    = _mm256_broadcast_ss(&a[0 * k + p]);
        ai2   = _mm256_broadcast_ss(&a[1 * k + p]);
        c0    = _mm256_fmadd_ps(ai, bp, c0);
        c1    = _mm256_fmadd_ps(ai2, bp, c1);

        ai    = _mm256_broadcast_ss(&a[2 * k + p]);
        ai2   = _mm256_broadcast_ss(&a[3 * k + p]);
        c2    = _mm256_fmadd_ps(ai, bp, c2);
        c3    = _mm256_fmadd_ps(ai2, bp, c3);

        ai    = _mm256_broadcast_ss(&a[4 * k + p]);
        ai2   = _mm256_broadcast_ss(&a[5 * k + p]);
        c4    = _mm256_fmadd_ps(ai, bp, c4);
        c5    = _mm256_fmadd_ps(ai2, bp, c5);

        ai    = _mm256_broadcast_ss(&a[6 * k + p]);
        ai2   = _mm256_broadcast_ss(&a[7 * k + p]);
        c6    = _mm256_fmadd_ps(ai, bp, c6);
        c7    = _mm256_fmadd_ps(ai2, bp, c7);

        ai    = _mm256_broadcast_ss(&a[8 * k + p]);
        ai2   = _mm256_broadcast_ss(&a[9 * k + p]);
        c8    = _mm256_fmadd_ps(ai, bp, c8);
        c9    = _mm256_fmadd_ps(ai2, bp, c9);

        ai    = _mm256_broadcast_ss(&a[10 * k + p]);
        ai2   = _mm256_broadcast_ss(&a[11 * k + p]);
        c10   = _mm256_fmadd_ps(ai, bp, c10);
        c11   = _mm256_fmadd_ps(ai2, bp, c11);
    }

    _mm256_storeu_ps(&c[0  * ld_c], c0);
    _mm256_storeu_ps(&c[1  * ld_c], c1);
    _mm256_storeu_ps(&c[2  * ld_c], c2);
    _mm256_storeu_ps(&c[3  * ld_c], c3);
    _mm256_storeu_ps(&c[4  * ld_c], c4);
    _mm256_storeu_ps(&c[5  * ld_c], c5);
    _mm256_storeu_ps(&c[6  * ld_c], c6);
    _mm256_storeu_ps(&c[7  * ld_c], c7);
    _mm256_storeu_ps(&c[8  * ld_c], c8);
    _mm256_storeu_ps(&c[9  * ld_c], c9);
    _mm256_storeu_ps(&c[10 * ld_c], c10);
    _mm256_storeu_ps(&c[11 * ld_c], c11);
}

float test_kernel(int k)
{
    int ld_c = 10 * NR;
    float *a = malloc(MR * k * sizeof(float));
    float *b = malloc(k * NR * sizeof(float));
    float *c = malloc(MR * ld_c * sizeof(float));
    float *c_check = malloc(MR * ld_c * sizeof(float));

    srand(time(NULL));

    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < k; j++) {
            a[i * k + j] = (float)rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < NR; j++) {
            b[i * NR + j] = (float)rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < ld_c; j++) {
            float x = (float)rand() / RAND_MAX;
            c[i * ld_c + j] = c_check[i * ld_c + j] = x;
        }
    }

    float *sub_c = c + 0;
    float *sub_c_check = c_check + 0;
    kernel(a, b, sub_c, k, ld_c);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                MR, NR, k, 1.0, a, k, b, NR, 1.0, sub_c_check, ld_c);

    float max_diff = 0;

    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < ld_c; j++) {
            float diff = fabs(c[i * ld_c + j] - c_check[i * ld_c + j]);
            max_diff = maxf(max_diff, diff);
        }
    }

    free(a);
    free(b);
    free(c);
    free(c_check);

    return max_diff;
}

double bench_kernel(int k, int iter)
{
    int ld_c = 10 * NR;
    float *a = calloc(MR * k, sizeof(float));
    float *b = calloc(k * NR, sizeof(float));
    float *c = calloc(MR * ld_c, sizeof(float));

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

    for (int t = 0; t < iter; t++) {
        kernel(a, b, c + 5, k, ld_c);
    }

    gettimeofday(&tv2, NULL);
    double duration = (double)(tv2.tv_usec - tv1.tv_usec) / 1e6 + 
                      (double)(tv2.tv_sec - tv1.tv_sec);

    free(a);
    free(b);
    free(c);

    return 2.0 * MR * k * NR * iter / duration / 1e9;
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
