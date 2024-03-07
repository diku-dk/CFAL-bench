#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <sys/time.h>

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

void FlashAttention(REAL *Q /* in N x d */, REAL *K /* in N x d */,
                    REAL *V /* in N x d */, REAL *O /* out N x d */,
                    int d, int N, int M)
{
    int Bc = ceildiv(M, sizeof(REAL) * d);
    int Br = min(Bc, d);
    int Tr = ceildiv(N, Br);
    int Tc = ceildiv(N, Bc);

    memset(O, 0, N * d * sizeof(REAL));
    REAL *l = calloc(N, sizeof(REAL));
    REAL *m = malloc(N * sizeof(REAL));
    REAL *mij = malloc(Br * sizeof(REAL));
    REAL *lij = malloc(Br * sizeof(REAL));
    REAL *mi_new = malloc(Br * sizeof(REAL));
    REAL *li_new = malloc(Br * sizeof(REAL));
    REAL *Pij = malloc(Bc * Br * sizeof(REAL));

    for (int i = 0; i < N; i++) {
        m[i] = -REAL_MAX;
    }

    for (int j = 0; j < Tc; j++) {
        REAL *Kj = K + j * Bc * d;
        REAL *Vj = V + j * Bc * d;
        for (int i = 0; i < Tr; i++) {
            REAL *Qi = Q + i * Br * d;
            REAL *Oi = O + i * Br * d;
            REAL *li = l + i * Br;
            REAL *mi = m + i * Br;

            matmulT(Qi, Kj, Pij, Br, d, Bc);

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

            cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       Br, d, Bc, 1.0, Pij, Bc, Vj, d, 1.0, Oi, d);

            for (int a = 0; a < Br; a++) {
                REAL scale = 1.0 / li_new[a];
                for (int b = 0; b < d; b++) {
                    Oi[a * d + b] *= scale;
                }
            }

            memcpy(mi, mi_new, Br * sizeof(REAL));
            memcpy(li, li_new, Br * sizeof(REAL));
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s d N M  Compute with matrices filled with ones\n", argv[0]);
        fprintf(stderr, "  Q, K, V are N x d matrices\n");
        fprintf(stderr, "  M is the size of a cache in bytes. ");
        fprintf(stderr, "Blocking is chosen such that half is filled\n");
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

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    FlashAttention(Q, K, V, O, d, N, M);
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

    fprintf(stderr,
            "Compute rate: %lf Gflops/s\n",
            2.0 * N * N * (d + 1) / duration / 1e9);

    free(Q);
    free(K);
    free(V);
    free(O);

    return EXIT_SUCCESS;
}
