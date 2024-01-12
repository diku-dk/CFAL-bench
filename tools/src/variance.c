#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#define WIDTH 32 /* SIMD width in bytes */
#define VEC (WIDTH / sizeof(double))

typedef double vd __attribute__ ((vector_size(WIDTH))); /* vector of doubles */
typedef size_t vs __attribute__ ((vector_size(WIDTH))); /* vector of size_ts */

struct stat {
    double mean;
    double stddev;
    size_t count;
};

/* Text input separated with newlines. */
struct stat Welford()
{
    size_t count = 0;
    double mean = 0;
    double m2 = 0;

    char *line = NULL;
    size_t len = 0;
    ssize_t nread;

    while ((nread = getline(&line, &len, stdin)) != -1) {
        double new_value = atof(line);
        count++;
        double delta_pre = new_value - mean;
        mean += delta_pre / count;
        double delta_post = new_value - mean;
        m2 += delta_pre * delta_post;
    }

    free(line);

    struct stat result = {mean, sqrt(m2 / count), count};
    return result;
}

/* Binary input */
struct stat WelfordBinary()
{
    double mean = 0;
    double m2 = 0;
    size_t count = 0;

    double new_value;
    size_t nread;
    while ((nread = fread(&new_value, sizeof(double), 1, stdin)) == 1) {
        count++;
        double delta_pre = new_value - mean;
        mean += delta_pre / count;
        double delta_post = new_value - mean;
        m2 += delta_pre * delta_post;
    }

    struct stat result = {mean, sqrt(m2 / count), count};
    return result;
}

/* Binary input Kahn */
struct stat WelfordBinaryKahn()
{
    double mean = 0;
    double mean_c = 0;
    double m2_c = 0;
    double m2 = 0;
    size_t count = 0;

    double new_value;
    size_t nread;
    while ((nread = fread(&new_value, sizeof(double), 1, stdin)) == 1) {
        count++;
        double delta_pre = new_value - mean;
        double mean_y = delta_pre / count - mean_c;
        double mean_t = mean + mean_y;
        mean_c = (mean_t - mean) - mean_y;
        mean = mean_t;
        double delta_post = new_value - mean;
        double m2_y = delta_pre * delta_post - m2_c;
        double m2_t = m2 + m2_y;
        m2_c = (m2_t - m2) - m2_y;
        m2 = m2_t;
    }

    struct stat result = {mean, sqrt(m2 / count), count};
    return result;
}

/* An even faster version taking binary input and using vectorisation. */
struct stat WelfordBinaryVec()
{
    vd means = {0, 0, 0, 0};
    vd m2s = {0, 0, 0, 0};
    vs counts = {0, 0, 0, 0};

    vd new_values;
    size_t nread;
    while ((nread = fread(&new_values, sizeof(double), VEC, stdin)) == VEC) {
        counts++;
        vd delta_pre = new_values - means;
        means += delta_pre / (vd)counts;
        vd delta_post = new_values - means;
        m2s += delta_pre * delta_post;
    }

    struct stat result = {means[0], sqrt(m2s[0] / counts[0]), counts[0]};
    return result;
}

int main(int argc, char **argv)
{
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "-b"))) {
        fprintf(stderr, "Usage: %s [-b] [-v]\nInput via stdin, "
                "default is text, with -b is in binary.", argv[0]);
        return EXIT_FAILURE;
    }

    struct stat result;

#ifdef BENCH
    clock_t start = clock();
#endif

    if (argc == 1) {
        result = Welford();
    } else {
//        result = WelfordBinaryVec();
//        result = WelfordBinaryKahn();
        result = WelfordBinary();
    }

#ifdef BENCH
    clock_t end = clock();
#endif

#ifdef BENCH
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "Processed %e doubles per second\n", result.count / time);
#endif

    printf("%.17g,%.17g\n", result.mean, result.stddev);

    return EXIT_SUCCESS;
}
