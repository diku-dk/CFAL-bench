#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Multiple of the vector width, as large as possible such that
 * VEC * NUM_X fits in cache. */
#define VEC 8

int    OUTER;
int    NUM_X;
int    NUM_Y;
int    NUM_T;
//double S0;
//double T;
//double ALPHA;
//double NU;
//double BETA;

#define OUTER 16
#define NUM_X 32
#define NUM_Y 256
#define NUM_T 256
#define S0 0.03
#define T 5.0
#define ALPHA 0.2
#define NU 0.6
#define BETA 0.5

void PrintArr(double *x, int m, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.17e ", x[i * n + j]);
        }
        printf("\n");
    }
}

void updateParams(double *X, double *Y, double t, double *VarX1, double *VarX2)
{
    for (int j = 0; j < NUM_Y; j++) {
        VarX1[j] = exp(2.0 * Y[j] - NU * NU * t);
    }

    for (int i = 0; i < NUM_X; i++) {
        VarX2[i] = pow(X[i], 2.0 * BETA);
    }
}

void initGrid(int *indX_o, int *indY_o, double *X, double *Y, 
              double *dx_o, double *dy_o)
{
    double stdX = 20.0 * ALPHA * S0 * sqrt(T);
    double dx = stdX / (double)NUM_X;
    int indX = (int)(S0 / dx);

    for (int i = 0; i < NUM_X; i++) {
        X[i] = (double)i * dx - (double)indX * dx + S0;
    }

    double stdY = 10.0 * NU * sqrt(T);
    double dy = stdY / NUM_Y;
    int indY = NUM_Y / 2;

    for (int i = 0; i < NUM_Y; i++) {
        Y[i] = (double)i * dy - (double)indY * dy + log(ALPHA);
    }

    *indX_o = indX;
    *indY_o = indY;
    *dx_o = dx;
    *dy_o = dy;
}

void tridag_v(double *a, double *b, double *y)
{
    //printf("a = \n");
    //for (int i = 0; i < NUM_X; i++) {
    //    for (int j = 0; j < VEC; j++) {
    //        printf("%e ", a[i * VEC + j]);
    //    }
    //}
    //printf("\n");

    //printf("b = \n");
    //for (int i = 0; i < NUM_X; i++) {
    //    for (int j = 0; j < VEC; j++) {
    //        printf("%e ", b[i * VEC + j]);
    //    }
    //}
    //printf("\n");

    //printf("Input:\n");
    //for (int i = 0; i < NUM_X; i++) {
    //    for (int j = 0; j < VEC; j++) {
    //        printf("%e ", y[i * VEC + j]);
    //    }
    //}
    //printf("\n");
    for (int j = 0; j < VEC; j++) {
        b[0 * VEC + j] = 1.0 / b[0 * VEC + j];
        y[0 * VEC + j] *= b[0 * VEC + j];
    }

    for (int j = 0; j < VEC; j++) {
        b[1 * VEC + j] = 1.0 / b[1 * VEC + j];
        y[1 * VEC + j] = b[1 * VEC + j] * (y[1 * VEC + j] - 
                                           a[1 * VEC + j] * y[0 * VEC + j]);
    }

    for (int i = 2; i < NUM_X; i++) {
        for (int j = 0; j < VEC; j++) {
            b[i * VEC + j] = 1.0 / (b[i * VEC + j] - 
                                a[i * VEC + j] * b[(i - 1) * VEC + j] * 
                                a[(i - 1) * VEC + j]);
            y[i * VEC + j] = b[i * VEC + j] * (y[i * VEC + j] - 
                                a[i * VEC + j] * y[(i - 1) * VEC + j]);
        }
    }

    //printf("Sweep 1:\n");
    //for (int i = 0; i < NUM_X; i++) {
    //    for (int j = 0; j < VEC; j++) {
    //        printf("%e ", y[i * VEC + j]);
    //    }
    //}
    //printf("\n");

    for (int i = NUM_X - 2; i >= 1; i--) {
        for (int j = 0; j < VEC; j++) {
            y[i * VEC + j] -= b[i * VEC + j] * a[i * VEC + j] * 
                                    y[(i + 1) * VEC + j];
        }
    }
    //printf("Output:\n");
    //for (int i = 0; i < NUM_X; i++) {
    //    for (int j = 0; j < VEC; j++) {
    //        printf("%e ", y[i * VEC + j]);
    //    }
    //}
    //printf("\n");
}

void toeplitz(double a, double b, double c, double *y, double *gamma)
{
    int n = NUM_Y;

//    printf("y[0] (input)\n");
//    PrintArr(y + 0 * NUM_X, 1, NUM_X);

    gamma[0] = 1.0 / b;
    for (int j = 0; j < NUM_X; j++) {
        y[0 * NUM_X + j] *= gamma[0];
    }
//    printf("y[0] (sweep 1)\n");
//    PrintArr(y + 0 * NUM_X, 1, NUM_X);

//    printf("y[1] (input) (%p)\n", y + 1 * NUM_X);
//    PrintArr(y + 1 * NUM_X, 1, NUM_X);
    gamma[1] = 1.0 / b;
    for (int j = 0; j < NUM_X; j++) {
        y[1 * NUM_X + j] = gamma[1] * (y[1 * NUM_X + j] - a * y[0 * NUM_X + j]);
    }
//    printf("y[1] (sweep 1)\n");
//    PrintArr(y + 1 * NUM_X, 1, NUM_X);

    for (int i = 2; i < n; i++) {
        gamma[i] = 1.0 / (b - a * gamma[i - 1] * c);
        for (int j = 0; j < NUM_X; j++) {
            y[i * NUM_X + j] = gamma[i] * (y[i * NUM_X + j] - 
                                        a * y[(i - 1) * NUM_X + j]);
        }
    }
//    printf("y[2] (sweep 1)\n");
//    PrintArr(y + 2 * NUM_X, 1, NUM_X);

    for (int i = n - 2; i >= 1; i--) {
        for (int j = 0; j < NUM_X; j++) {
            y[i * NUM_X + j] -= gamma[i] * c * y[(i + 1) * NUM_X + j];
        }
    }
}

double L2(double *x, int elem)
{
    double sum = 0.0;
    for (int i = 0; i < elem; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

double rollback(double *ResultE, double dtInv, double dx, double *X, double *Y,
                double *VarX1, double *VarX2, double dy, double VarY,
                double *V, double *U, double *a, double *b, double *gamma,
                double *uu)
{
    printf("dtInv = %.17e\n", dtInv);

    /* explicit y */
    for (int i = 0; i < NUM_X; i++) {
        V[0 * NUM_X + i] = 0;
    }
    for (int j = 1; j < NUM_Y - 1; j++) {
        for (int i = 0; i < NUM_X; i++) {
            V[j * NUM_X + i] = VarY / 4 * 
                        (ResultE[(j - 1) * NUM_X + i] / (dy * dy) +
                  -2.0 * ResultE[ j      * NUM_X + i] / (dy * dy) +
                         ResultE[(j + 1) * NUM_X + i] / (dy * dy));
        }
    }
    for (int i = 0; i < NUM_X; i++) {
        V[(NUM_Y - 1) * NUM_X + i] = 0;
    }
    printf("L2(V) = %.17e\n", L2(V, NUM_X * NUM_Y));

    /* explicit x */
    for (int j = 0; j < NUM_Y; j++) {
        U[j * NUM_X + 0] = dtInv * ResultE[j * NUM_X + 0];
        for (int i = 1; i < NUM_X - 1; i++) {
            /**
             * For the first pass, ResultE[j, i] depends only on i, so
             * algebraically this simplifies to dtInv * ResultE[j * NUM_X + i].
             * Printing shows that the rows are not exactly equal though, even
             * when doing everything IEEE compliant. The SaC and C code have
             * exactly the same dtInv, ResultE, VarX1, VarX2, but U differs
             * slightly (about 1e-13) (and both are unequal to the true, infinite
             * precision answer, as it should be dtInv * ResultE). 
             * However, where the SaC converges, this version
             * diverges.
             **/
            U[j * NUM_X + i] = dtInv * ResultE[j * NUM_X + i] +
                                  VarX1[j] * VarX2[i] / 4.0 * (
                                       ResultE[j * NUM_X + i - 1] / (dx * dx) +
                               -2.0 *  ResultE[j * NUM_X + i    ] / (dx * dx) +
                                       ResultE[j * NUM_X + i + 1] / (dx * dx)
                               );
        }
        U[j * NUM_X + NUM_X - 1] = dtInv * ResultE[j * NUM_X + NUM_X - 1];
    }
    printf("L2(U) = %.17e\n", L2(U, NUM_X * NUM_Y));
    PrintArr(U, NUM_Y, NUM_X);
//    PrintArr(U, NUM_Y, NUM_X);

    for (int j = 0; j < NUM_Y; j++) {
        for (int i = 0; i < NUM_X; i++) {
            U[j * NUM_X + i] = U[j * NUM_X + i] + 2.0 * V[j * NUM_X + i];
        }
    }
    printf("L2(U + 2V) = %.17e\n", L2(U, NUM_X * NUM_Y));

    /* implicit x (takes 60% of the total time, a significant amount of time
     * is in computing a and b) */

    for (int j = 0; j < NUM_Y; j++) {
        VarX1[j] *= -0.25 / dtInv / (dx * dx);
    }
    for (int j = 0; j < NUM_Y; j += VEC) {
        for (int k = 0; k < VEC; k++) {
            a[k] = 0;
            b[k] = 1;
            for (int i = 1; i < NUM_X - 1; i++) {
                a[i * VEC + k] = VarX1[j + k] * VarX2[i];
                b[i * VEC + k] = 1.0 - 2.0 * a[i * VEC + k];
            }
            a[(NUM_X - 1) * VEC + k] = 0;
            b[(NUM_X - 1) * VEC + k] = 1;
        }

        for (int k = 0; k < VEC; k++) {
            for (int i = 0; i < NUM_X; i++) {
                uu[i * VEC + k] = U[(j + k) * NUM_X + i];
            }
        }

        tridag_v(a, b, uu);

        for (int k = 0; k < VEC; k++) {
            for (int i = 0; i < NUM_X; i++) {
                U[(j + k) * NUM_X + i] = uu[i * VEC + k];
            }
        }
    }

    printf("L2(implicit U) = %.17e\n", L2(U, NUM_X * NUM_Y));

//    printf("U[1]\n");
//    PrintArr(U + 1 * NUM_X, 1, NUM_X);
//    printf("V[1]\n");
//    PrintArr(V + 1 * NUM_X, 1, NUM_X);

//    for (int i = 0; i < NUM_X; i++) {
//        printf("%lf ", U[2 * NUM_X + i]);
//    }
//    printf("\n");

    /* implicit y */
    for (int j = 0; j < NUM_Y; j++) {
        for (int i = 0; i < NUM_X; i++) {
            ResultE[j * NUM_X + i] = U[j * NUM_X + i] - V[j * NUM_X + i];
        }
    }
//    PrintArr(ResultE + 1 * NUM_X, 1, NUM_X);

    double a_scalar = VarY * -0.25 / (dy * dy);
    toeplitz(a_scalar, dtInv - 2 * a_scalar, a_scalar, ResultE, gamma);
    printf("L2(implicit y) = %.17e\n", L2(ResultE, NUM_X * NUM_Y));
}

double value(double strike, int indX, int indY, double *X, double *Y, 
             double *VarX1, double *VarX2, 
             double dx, double dy)
{
    /* Both NUM_Y x NUM_X */
    double *ResultE = malloc(NUM_X * NUM_Y * sizeof(double));
    double *V       = malloc(NUM_X * NUM_Y * sizeof(double));
    double *U       = malloc(NUM_X * NUM_Y * sizeof(double));
    double *a       = malloc(VEC * NUM_X * sizeof(double));
    double *b       = malloc(VEC * NUM_X * sizeof(double));
    double *gamma   = malloc(NUM_Y * sizeof(double));
    double *uu      = malloc(VEC * NUM_X * sizeof(double));

    printf("strike = %.17e\n", strike);
    for (int j = 0; j < NUM_Y; j++) {
        for (int i = 0; i < NUM_X; i++) {
            ResultE[j * NUM_X + i] = (X[i] - strike >= 0) ? X[i] - strike : 0;
        }
    }
    printf("\n");
    PrintArr(ResultE, NUM_Y, NUM_X);

    for (int t = NUM_T - 2; t >= 0; t--) {
        rollback(ResultE, (double)(NUM_T - 1) / T, dx, X, Y, 
                 VarX1, VarX2, dy, NU * NU, V, U, a, b, gamma, uu);
        for (int j = 0; j < NUM_X; j++) {
            VarX2[j] *= exp(NU * NU * T / (NUM_T - 1));
        }
    }

    double res = ResultE[indY * NUM_X + indX];
    free(ResultE);
    free(V);
    free(U);
    free(a);
    free(b);
    free(uu);
    free(gamma);
    return res;
}

int main(int argc, char **argv)
{
    if (argc != 10) {
        printf("Usage: %s OUTER NUM_X NUM_Y NUM_T S0 T ALPHA NU BETA\n", 
               argv[0]);
        return EXIT_FAILURE;
    }

//    OUTER = atoi(argv[1]);
//    NUM_X = atoi(argv[2]);
//    NUM_Y = atoi(argv[3]);
//    NUM_T = atoi(argv[4]);
//    S0    = atof(argv[5]);
//    T     = atof(argv[6]);
//    ALPHA = atof(argv[7]);
//    NU    = atof(argv[8]);
//    BETA  = atof(argv[9]);

    int indX, indY;
    double dx, dy;
    double *X     = malloc(NUM_X * sizeof(double));
    double *Y     = malloc(NUM_Y * sizeof(double));
    double *VarX1 = malloc(NUM_Y * sizeof(double));
    double *VarX2 = malloc(NUM_X * sizeof(double));

    initGrid(&indX, &indY, X, Y, &dx, &dy);
    PrintArr(X, 1, NUM_X);
    PrintArr(Y, 1, NUM_Y);
    updateParams(X, Y, (double)(NUM_T - 2) * T / (NUM_T - 1), VarX1, VarX2);
    PrintArr(VarX1, 1, NUM_Y);
    PrintArr(VarX2, 1, NUM_X);

    double result[OUTER];
    for (int i = 0; i < OUTER; i++) {
        result[i] = value((double)i / 1000.0, indX, indY, X, Y, 
                          VarX1, VarX2, dx, dy);
    }

    for (int i = 0; i < OUTER; i++) {
        printf("%lf ", result[i]);
    }
    printf("\n");

    free(X);
    free(Y);
    free(VarX1);
    free(VarX2);

    return EXIT_SUCCESS;
}
