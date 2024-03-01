#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


enum Mode {
    MODE_ONE,
    MODE_RANDOM,
};

float rand_float(void) {
    // random(3) returns a random number in [0, 2^31 - 1].
    return (float)random() / (float)INT32_MAX * 2 - 1;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s d N -one        Generate matrices filled with ones\n", argv[0]);
        fprintf(stderr, "  %s d N -random     Generate random matrices\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int d = atoi(argv[1]);
    const int N = atoi(argv[2]);

    enum Mode mode;
    if (strcmp(argv[3], "-one") == 0) {
        mode = MODE_ONE;
    } else if (strcmp(argv[3], "-random") == 0) {
        mode = MODE_RANDOM;
    } else {
        fprintf(stderr, "Invalid mode\n");
        return EXIT_FAILURE;
    }

    printf("%d %d\n\n", d, N);

    // generate Q, K, V, all of dimensions N x d (rows x cols).

    for (int mat_i = 0; mat_i < 3; mat_i++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < d; j++) {
                if (j > 0) putchar(' ');
                switch (mode) {
                    case MODE_ONE: printf("1"); break;
                    case MODE_RANDOM: printf("%f", rand_float()); break;
                }
            }
            putchar('\n');
        }
        putchar('\n');
    }
}
