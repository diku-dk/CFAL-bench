#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv)
{
    if (argc == 1 || argc > 3) {
        fprintf(stderr, "Usage: %s n for n numbers in text "
                "or %s -b n for n numbers in binary\n", argv[0], argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 2) {
        int elems = atoi(argv[1]);
        for (int i = 0; i < elems; i++) {
            printf("%lf\n", (double)i);
        }
    } else {
        int elems;
        if (!strcmp(argv[1], "-b")) {
            elems = atoi(argv[2]);
        } else if (!strcmp(argv[2], "-b")) {
            elems = atoi(argv[1]);
        } else {
            fprintf(stderr, "Usage: %s n for n numbers in text "
                "or %s -b n for n numbers in binary\n", argv[0], argv[0]);
            return EXIT_FAILURE;
        }

        srand(time(NULL));

        double number[1];
        for (int i = 0; i < elems; i++) {
            number[0] = (double)rand() / RAND_MAX;
            fwrite(number, sizeof(double), 1, stdout);
        }
    }

    return EXIT_SUCCESS;
}
