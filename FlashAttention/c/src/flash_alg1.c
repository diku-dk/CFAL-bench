/***********************************************************
 * A faithful implementation of Algorithm 1 from the paper *
 * ======================================================= *
 * Implementor: Aaron W. Hsu <arcfide@sacrideo.us>         *
 * Date: March 2024                                        *
 ***********************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

float
L2(float *x, size_t c)
{
	float sum;
	
	sum = 0;
	
	for (size_t i = 0; i < c; i++)
		sum += x[i] * x[i];
	
	return sqrt(sum);
}

int
flash_attention(float *O, float *Q, float *K, float *V, int N, int d, int M)
{
	float *l, *m, *Pij;
	int Br, Bc, Tr, Tc;
	
	Bc = M / (4 * d);
	Br = d < Bc ? d : Bc;
	Tr = N / Br;
	Tc = N / Bc;
	
	if ((l = calloc(N, sizeof(float))) == NULL)
		return 1;
	
	if ((m = calloc(N, sizeof(float))) == NULL)
		return 1;
	
	if ((Pij = calloc(Bc, sizeof(float))) == NULL)
		return 1;
	
	for (int i = 0; i < N; i++)
		m[i] = -INFINITY;
	
	for (int j = 0; j < Tc; j++) {
		float *Kj, *Vj;
		
		Kj = K + j * Bc * d;
		Vj = V + j * Bc * d;
		
		for (int i = 0; i < Tr; i++) {
			float *Oi, *Qi, *mi, *li;
			
			Oi = O + i * Br * d;
			Qi = Q + i * Br * d;
			mi = m + i * Br;
			li = l + i * Br;
			
			for (int ii = 0; ii < Br; ii++) {
				float sum, max, li_new, mi_new, eij, eli;
				
				max = -INFINITY;

				for (int jj = 0; jj < Bc; jj++) {
					sum = 0;
					
					for (int kk = 0; kk < d; kk++) {
						float x, y;
						
						x = Qi[ii * d + kk];
						y = Kj[jj * d + kk];
						
						sum += x * y;
					}
					
					Pij[jj] = sum;
					max = sum > max ? sum : max;
				}
				
				sum = 0;

				for (int jj = 0; jj < Bc; jj++) {
					float *x = &Pij[jj];
					
					sum += *x = exp(*x - max);
				}
				
				mi_new = mi[ii] > max ? mi[ii] : max;
				eij = exp(max - mi_new);
				eli = li[ii] * exp(mi[ii] - mi_new);
				li_new = eli + sum * eij;
				
				li[ii] = li_new;
				mi[ii] = mi_new;
				
				for (int kk = 0; kk < d; kk++)
					Oi[ii * d + kk] *= eli;
				
				for (int jj = 0; jj < Bc; jj++) {
					float x = eij * Pij[jj];

					for (int kk = 0; kk < d; kk++) {
						int ik = ii * d + kk;
						int jk = jj * d + kk;
						
						Oi[ik] += x * Vj[jk];
					}
				}
				
				for (int kk = 0; kk < d; kk++)
					Oi[ii * d + kk] /= li_new;
			}
		}
	}
	
	free(l);
	free(m);
	free(Pij);
	
	return 0;
}

int main(int argc, char **argv)
{
	struct timeval ts, te;
	size_t cnt;
	float *Q, *K, *V, *O, dur;
	int N, d, M;

	if (argc != 3 && argc != 4) {
		fprintf(stderr, "Usage:\n");
		fprintf(stderr, "  %s d N M  Compute with matrices filled with ones\n", argv[0]);
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

	if (io_arrays) {
		M = atoi(argv[1]);
		scanf("%d %d", &d, &N);
	} else {
		d = atoi(argv[1]);
		N = atoi(argv[2]);
		M = atoi(argv[3]);
	}

    if (N % d != 0) {
        fprintf(stderr, "d must divide N\n");
        return EXIT_FAILURE;
    }

	cnt = N * d;

	fprintf(stderr, "Initializing data...");

	if ((Q = calloc(cnt, sizeof(float))) == NULL)
		goto mem_failure;

	if ((K = calloc(cnt, sizeof(float))) == NULL)
		goto mem_failure;

	if ((V = calloc(cnt, sizeof(float))) == NULL)
		goto mem_failure;

	if ((O = calloc(cnt, sizeof(float))) == NULL)
		goto mem_failure;
	
	if (io_arrays) {
		for (int i = 0; i < d * N; i++) scanf("%f", &Q[i]);
		for (int i = 0; i < d * N; i++) scanf("%f", &K[i]);
		for (int i = 0; i < d * N; i++) scanf("%f", &V[i]);
	} else {
		for (size_t i = 0; i < cnt; i++) {
			Q[i] = K[i] = V[i] = 1;
		}
	}
	
	fprintf(stderr, "done.\n");
	fprintf(stderr, "Warming up...");
	
	if (flash_attention(O, Q, K, V, N, d, M))
		goto attn_failure;
	
	fprintf(stderr, "done.\n");	
	fprintf(stderr, "Running flash_attention...");
	
	gettimeofday(&ts, NULL);
	
	if (flash_attention(O, Q, K, V, N, d, M))
		goto attn_failure;
	
	gettimeofday(&te, NULL);
	
	fprintf(stderr, "done.\n");
	
	dur = (double)(te.tv_usec - ts.tv_usec) / 1e6 + 
		      (double)(te.tv_sec - ts.tv_sec);

	if (io_arrays) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < d; j++) {
				if (j > 0) putchar(' ');
				printf("%f", O[d * i + j]);
			}
			putchar('\n');
		}
	} else {
		fprintf(stderr, "L2 norm is %lf (should be %lf)\n", L2(O, cnt), sqrt(cnt));
	}

    /* QK^t is 2N^2d flops, so is PV. softmax(S) (row-wise)
     * exp(S[i]) / sum_j exp(P[i, j] - max(P[i])) 
     * is N * (N + 4N) = 5 N^2 flops, but exp is more expensive. */
    fprintf(stderr,
            "Compute rate: %lf Gflops/s\n", 
            (4.0 * d + 5.0) * N * N / dur / 1e9);

	free(Q);
	free(K);
	free(V);
	free(O);
	
	return EXIT_SUCCESS;
	
mem_failure:
	printf("Failed to allocate memory.\n");
	return EXIT_FAILURE;
	
attn_failure:
	printf("Call to flash_attention failed.\n");
	return EXIT_FAILURE;
}
