#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <cblas.h>
#include <sys/time.h>

#define REAL float
#define exp expf
#define cblas_gemm cblas_sgemm

void SetOne(REAL *x, int elems)
{
    for (int i = 0; i < elems; i++) {
        x[i] = 1;
    }
}

void softmax(REAL *x, int hei, int wid)
{
    for (int i = 0; i < hei; i++) {
        REAL max = 0;
        for (int j = 0; j < wid; j++) {
            if (x[i * hei + j] > max) max = x[i * hei + j];
        }
        REAL total = 0;
        for (int j = 0; j < wid; j++) {
            const REAL fx_i = exp(x[i * hei + j] - max);
            x[i * hei + j] = fx_i;
            total += fx_i;
        }
        for (int j = 0; j < wid; j++) {
            x[i * hei + j] /= total;
        }
    }
}

/* a is m x k, b is k x n */
void matmul(REAL *a, REAL *b, REAL *c, int m, int k, int n)
{
    cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               m, n, k, 1.0, a, k, b, n, 0.0, c, n);
}

/* A is m x k, b is n x k */
void matmulT(REAL *a, REAL *b, REAL *c, int m, int k, int n)
{
    cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
               m, n, k, 1.0, a, k, b, k, 0.0, c, n);
}

REAL L2(REAL *x, int elems)
{
    REAL sum = 0;
    for (int i = 0; i < elems; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

void Attention(REAL *Q, REAL *K, REAL *V, REAL *S, REAL *O, 
               int d, int N)
{
    // Q, K, V :: N x d
    // S = Q * K^T :: N x N
    matmulT(Q, K, S, N, d, N);
    // S' := P = softmax(S) :: N x N
    softmax(S, N, N);  // P is in S now
    // O = P * V :: N x d
    matmul(S, V, O, N, N, d);
}

int main(int argc, char **argv)
{
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s d N    Compute with matrices filled with ones\n", argv[0]);
        fprintf(stderr, "  %s -io    Read matrices from stdin and write O to stdout\n", argv[0]);
        return EXIT_FAILURE;
    }

    bool io_arrays = false;
    if (argc == 2) {
        if (strcmp(argv[1], "-io") != 0) {
            fprintf(stderr, "Invalid argument '%s'\n", argv[1]);
            return EXIT_FAILURE;
        }
        io_arrays = true;
    }

    int d, N;
    if (io_arrays) {
        scanf("%d %d", &d, &N);
    } else {
        d = atoi(argv[1]);
        N = atoi(argv[2]);
    }

    if (N % d != 0) {
        fprintf(stderr, "d must divide N\n");
        return EXIT_FAILURE;
    }

    REAL *Q = malloc(d * N * sizeof(REAL));
    REAL *K = malloc(d * N * sizeof(REAL));
    REAL *V = malloc(d * N * sizeof(REAL));
    REAL *S = malloc(N * N * sizeof(REAL));
    REAL *O = malloc(d * N * sizeof(REAL));

    if (io_arrays) {
        for (int i = 0; i < d * N; i++) scanf("%f", &Q[i]);
        for (int i = 0; i < d * N; i++) scanf("%f", &K[i]);
        for (int i = 0; i < d * N; i++) scanf("%f", &V[i]);
    } else {
        SetOne(Q, d * N);
        SetOne(K, d * N);
        SetOne(V, d * N);
    }

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    Attention(Q, K, V, S, O, d, N);
    gettimeofday(&tv2, NULL);
    double duration = (double)(tv2.tv_usec - tv1.tv_usec) / 1e6 + 
                      (double)(tv2.tv_sec - tv1.tv_sec);

    if (io_arrays) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < d; j++) {
                if (j > 0) putchar(' ');
                printf("%f", O[d * i + j]);
            }
            putchar('\n');
        }
    } else {
        fprintf(stderr, "L2 norm is %lf (should be %lf)\n", L2(O, d * N), sqrt(N * d));
    }

    /* QK^t is 2N^2d flops, so is PV. softmax(S) (row-wise)
     * exp(S[i]) / sum_j exp(P[i, j] - max(P[i])) 
     * is N * (N + 4N) = 5 N^2 flops, but exp is more expensive. */
    fprintf(stderr,
            "Compute rate: %lf Gflops/s\n", 
            (4.0 * d + 5.0) * N * N / duration / 1e9);

    free(Q);
    free(K);
    free(V);
    free(S);
    free(O);

    return EXIT_SUCCESS;
}
