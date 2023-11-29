/* TODO numerical stability */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <sys/time.h>
#include <omp.h>

#define REAL float
#define exp expf
#define cblas_gemm cblas_sgemm

void SetOne(REAL *x, int elems)
{
    for (int i = 0; i < elems; i++) {
        x[i] = 1;
    }
}

void scale(REAL *x, int m, int n)
{
    for (int i = 0; i < m; i++) {
        REAL weight = 0;
        for (int j = 0; j < n; j++) {
            weight += x[i * n + j];
        }
        weight = 1 / weight;
        for (int j = 0; j < n; j++) {
            x[i * n + j] *= weight;
        }
    }
}

void stabilize(REAL *x, int m, int n)
{
    for (int i = 0; i < m; i++) {
        REAL maximum = x[i * n + 0];
        for (int j = 0; j < n; j++) {
            if (x[i * n + j] > maximum) {
                maximum = x[i * n + j];
            }
        }
        for (int j = 0; j < n; j++) {
            x[i * n + j] -= maximum;
        }
    }
}

void exp_arr(REAL *x, int size)
{
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
    }
}

/* a is m x k, b is k x n */
void matmul(REAL *a, REAL *b, REAL *c, int m, int k, int n)
{
    cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               n, m, k, 1.0, a, k, b, n, 0.0, c, n);
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

void FlashAttention(REAL *Q, REAL *K, REAL *V, REAL *O, REAL *P_blocks, 
                    int d, int N)
{
    #pragma omp parallel 
    {
        REAL *P_block = P_blocks + d * N * omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < N / d; i++) {
            /* P_block = matmul(Q[i], K^t). Q[i] is d x d, K is N x d. */
            matmulT(Q + i * d * d, K, P_block, d, d, N);
            stabilize(P_block, d, N);
            exp_arr(P_block, N * d);
            scale(P_block, d, N);
            /* P_block is d x N, V is N x d. */
            matmul(P_block, V, O + i * d * d, d, N, d);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: d N\n");
        return EXIT_FAILURE;
    }

    int d = atoi(argv[1]);
    int N = atoi(argv[2]);

    if (N % d != 0) {
        printf("d must divide N\n");
        return EXIT_FAILURE;
    }

    /* Don't use more than 1 thread for the gemm call */
    omp_set_nested(0);

    int num_threads;

    #pragma omp parallel 
    {
        #pragma omp master
        num_threads = omp_get_num_threads();
    }
    printf("num_threads = %d\n", num_threads);

    REAL *Q = malloc(d * N * sizeof(REAL));
    REAL *K = malloc(d * N * sizeof(REAL));
    REAL *V = malloc(d * N * sizeof(REAL));
    REAL *O = malloc(d * N * sizeof(REAL));
    REAL *P_blocks = malloc(d * N * sizeof(REAL) * num_threads);

    SetOne(Q, d * N);
    SetOne(K, d * N);
    SetOne(V, d * N);

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    FlashAttention(Q, K, V, O, P_blocks, d, N);
    gettimeofday(&tv2, NULL);
    double duration = (double)(tv2.tv_usec - tv1.tv_usec) / 1e6 + 
                      (double)(tv2.tv_sec - tv1.tv_sec);

    printf("L2 norm is %lf (should be %lf)\n", L2(O, d * N), sqrt(N * d));
    printf("Compute rate: %lf Gflops/s\n", 
            2.0 * N * N * (d + 1) / duration / 1e9);

    free(Q);
    free(K);
    free(V);
    free(O);
    free(P_blocks);

    return EXIT_SUCCESS;
}
