/**
 * Custom matmul implementation, multithreaded.
 **/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <sys/time.h>
#include <float.h>
#include <omp.h>
#include "matmul.h"

#define REAL float
#define exp expf
#define cblas_gemm cblas_sgemm
#define REAL_MAX FLT_MAX

void SetOne(REAL *x, int elems)
{
    for (int i = 0; i < elems; i++) {
        x[i] = 1;
    }
}

REAL L2(REAL *x, int elems)
{
    REAL sum = 0;
    for (int i = 0; i < elems; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

inline
int ceildiv(int a, int b)
{
    return (a + b - 1) / b;
}

inline
int min(int a, int b)
{
    return (a < b) ? a : b;
}

inline
REAL maxr(REAL a, REAL b)
{
    return (a > b) ? a : b;
}

void rowmax(REAL *Sij /* Br x Bc in */,  REAL *mij /* Br, out */,
            int Br, int Bc)
{
    for (int i = 0; i < Br; i++) {
        REAL res = Sij[i * Bc];
        for (int j = 1; j < Bc; j++) {
            res = maxr(Sij[i * Bc + j], res);
        }
        mij[i] = res;
    }
}

void rowsum(REAL *Pij /* Br x Bc in */,  REAL *lij /* Br, out */,
            int Br, int Bc)
{
    for (int i = 0; i < Br; i++) {
        REAL res = 0;
        for (int j = 0; j < Bc; j++) {
            res += Pij[i * Bc + j];
        }
        lij[i] = res;
    }
}

// /* a is m x k, b is k x n */
// void matmul(REAL *a, REAL *b, REAL *c, int m, int k, int n)
// {
//     cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//                m, n, k, 1.0, a, k, b, n, 1.0, c, n);
// }

// /* A is m x k, b is n x k */
// void matmulT(REAL *a, REAL *b, REAL *c, int m, int k, int n)
// {
//     cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
//                m, n, k, 1.0, a, k, b, k, 0.0, c, n);
// }

void FlashAttention(REAL *Q /* in N x d */, REAL *K /* in N x d */,
                    REAL *V /* in N x d */, REAL *O /* out N x d */,
                    int d, int N, int M)
{
    // int Bc = ceildiv(M, sizeof(REAL) * d);
    // int Br = min(Bc, d);
    int Bc = M;
    int Br = M;
    int Tr = ceildiv(N, Br);
    int Tc = ceildiv(N, Bc);

    if (Br % 2 != 0 || Bc % KC != 0 || d % NR != 0) {
        fprintf(stderr, "Br = %d, Bc = %d, d = %d not good\n", Br, Bc, d);
        abort();
    }

    #pragma omp parallel
    {
        REAL *Kjt    = malloc(d * Bc * sizeof(REAL));
        REAL *l      = malloc(Br * sizeof(REAL));
        REAL *m      = malloc(Br * sizeof(REAL));
        REAL *mij    = malloc(Br * sizeof(REAL));
        REAL *lij    = malloc(Br * sizeof(REAL));
        REAL *mi_new = malloc(Br * sizeof(REAL));
        REAL *li_new = malloc(Br * sizeof(REAL));
        REAL *Pij    = malloc(Bc * Br * sizeof(REAL));
        REAL *Oi = malloc(Br * d * sizeof(REAL));

        #pragma omp for
        for (int i = 0; i < Tr; i++) {
            
            REAL *Qi = Q + i * Br * d;
            memset(Oi, 0, Br * d * sizeof(REAL));
            for (int idx = 0; idx < Br; idx++) {
                m[idx] = -REAL_MAX;
                l[idx] = 0;
            }
            REAL *li = l;
            REAL *mi = m;

            for (int j = 0; j < Tc; j++) {
                REAL *Kj = K + j * Bc * d;
                REAL *Vj = V + j * Bc * d;

                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < Bc; b++) {
                        Kjt[a * Bc + b] = Kj[b * d + a];
                    }
                }

                memset(Pij, 0, Bc * Br * sizeof(REAL));
                matmul(Qi, Kjt, Pij, Br, d, Bc);
                // matmulT(Qi, Kj, Pij, Br, d, Bc);

                rowmax(Pij, mij, Br, Bc);

                for (int a = 0; a < Br; a++) {
                    for (int b = 0; b < Bc; b++) {
                        Pij[a * Bc + b] = exp(Pij[a * Bc + b] - mij[a]);
                    }
                }

                rowsum(Pij, lij, Br, Bc);

                for (int a = 0; a < Br; a++) {
                    mi_new[a] = maxr(mi[a], mij[a]);
                    li_new[a] = exp(mi[a] - mi_new[a]) * li[a] +
                                exp(mij[a] - mi_new[a]) * lij[a];
                }

                for (int a = 0; a < Br; a++) {
                    for (int b = 0; b < d; b++) {
                        Oi[a * d + b] *= li[a] * exp(mi[a] - mi_new[a]);
                    }
                }

                for (int a = 0; a < Br; a++) {
                    REAL scale = exp(mij[a] - mi_new[a]);
                    for (int b = 0; b < Bc; b++) {
                        Pij[a * Bc + b] *= scale;
                    }
                }

                matmul(Pij, Vj, Oi, Br, Bc, d);

                for (int a = 0; a < Br; a++) {
                    REAL scale = 1.0 / li_new[a];
                    for (int b = 0; b < d; b++) {
                        Oi[a * d + b] *= scale;
                    }
                }

                memcpy(mi, mi_new, Br * sizeof(REAL));
                memcpy(li, li_new, Br * sizeof(REAL));
            }

            memcpy(O + i * Br * d, Oi, Br * d * sizeof(REAL));
        }

        free(Kjt);
        free(l);
        free(m);
        free(mij);
        free(lij);
        free(mi_new);
        free(li_new);
        free(Pij);
        free(Oi);
    }
}

int main(int argc, char **argv)
{
    // openblas_set_num_threads(1);

    if (argc != 3 && argc != 4) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s d N M  Compute with matrices filled with ones\n", argv[0]);
        fprintf(stderr, "  Q, K, V are N x d matrices\n");
        // fprintf(stderr, "  M is the size of a cache in bytes. ");
        fprintf(stderr, "  M is the blovk size. ");
        // fprintf(stderr, "Blocking is chosen such that half is filled\n");
		fprintf(stderr, "  %s M -io  Read matrices from stdin and write O to stdout\n", argv[0]);
        return EXIT_FAILURE;
    }

    bool io_arrays = false;
    if (argc == 3) {
        if (strcmp(argv[2], "-io") != 0) {
            fprintf(stderr, "Invalid argument '%s'\n", argv[1]);
            return EXIT_FAILURE;
        }
        io_arrays = true;
    }

    int d, N, M;
    if (io_arrays) {
		M = atoi(argv[1]);
		scanf("%d %d", &d, &N);
    } else {
        d = atoi(argv[1]);
        N = atoi(argv[2]);
        M = atoi(argv[3]);
    }

    REAL *Q = malloc(d * N * sizeof(REAL));
    REAL *K = malloc(d * N * sizeof(REAL));
    REAL *V = malloc(d * N * sizeof(REAL));
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

    // Warm-up and validation run
    FlashAttention(Q, K, V, O, d, N, M);

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

    struct timeval tv1, tv2;
    double runtimes[10];
    for (int r = 0; r < 10; r++) {
        gettimeofday(&tv1, NULL);
        FlashAttention(Q, K, V, O, d, N, M);
        gettimeofday(&tv2, NULL);
        runtimes[r] = (double)(tv2.tv_usec - tv1.tv_usec) / 1e6 +
                      (double)(tv2.tv_sec - tv1.tv_sec);
    }
    double mean = 0.0;
    for (int r = 0; r < 10; r++) {
        mean += runtimes[r];
    }
    mean /= 10;
    double std = 0.0;
    for (int r = 0; r < 10; r++) {
        std += (runtimes[r] - mean) * (runtimes[r] - mean);
    }
    std = sqrt(std / 10);
    double percent = 100.0 * std / mean;

    /* QK^t is 2N^2d flops, so is PV. softmax(S) (row-wise)
     * exp(S[i]) / sum_j exp(P[i, j] - max(P[i])) 
     * is N * (N + 4N) = 5 N^2 flops, but exp is more expensive. */
    fprintf(stderr,
            "Runtime: %lf s (%lf stdev - %.2lf%%)\n",
            mean, std, percent);
    fprintf(stderr,
            "Compute rate: %lf Gflops/s\n", 
            (4.0 * d + 5.0) * N * N / mean / 1e9);

    free(Q);
    free(K);
    free(V);
    free(O);

    return EXIT_SUCCESS;
}
