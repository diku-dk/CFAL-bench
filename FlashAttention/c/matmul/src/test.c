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
 *                    112 Gflops/s on bin/matmul 256 64 256 100000
 *
 * Peak performance is 128 Gflops/s, so this is about 90%
 * TODO: I don't see why this cannot be 98% as well. Probably messed something
 * up in the streaming from L1. Can prefetch handle the strided access?
 **/

#include "../include/matmul.h"
#include <cblas.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

float maxf(float a, float b)
{
    return (a > b) ? a : b;
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
