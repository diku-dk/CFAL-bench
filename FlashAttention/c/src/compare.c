#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define REAL float


int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "Usage: d N <out1.txt> <out2.txt>\n");
        return EXIT_FAILURE;
    }

    const int d = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const char *fname1 = argv[3];
    const char *fname2 = argv[4];

    FILE *f1 = fopen(fname1, "r");
    FILE *f2 = fopen(fname2, "r");
    if (!f1 || !f2) {
        fprintf(stderr, "Cannot open input files\n");
        return EXIT_FAILURE;
    }

    REAL maxdiff = 0;  // max_i |a_i - b_i|
    REAL diffsum = 0;  // sum_i |a_i - b_i|

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            REAL a, b;
            fscanf(f1, "%f", &a);
            fscanf(f2, "%f", &b);
            maxdiff += fabs(a - b);
            diffsum += fabs(a - b);
        }
    }

    printf("Maximum absolute difference: %f\n", maxdiff);
    printf("Average absolute difference: %f\n", diffsum / (N * d));
}
