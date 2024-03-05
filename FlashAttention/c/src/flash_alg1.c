/***********************************************************
 * A faithful implementation of Algorithm 1 from the paper *
 * ======================================================= *
 * Implementor: Aaron W. Hsu <arcfide@sacrideo.us>         *
 * Date: March 2024                                        *
 ***********************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

double
L2(double *x, size_t c)
{
	double sum;
	
	sum = 0;
	
	for (size_t i = 0; i < c; i++)
		sum += x[i] * x[i];
	
	return sqrt(sum);
}

int
flash_attention(double *O, double *Q, double *K, double *V, int N, int d, int M)
{
	double *l, *m, *Pij;
	int Br, Bc, Tr, Tc;
	
	Bc = M / (4 * d);
	Br = d < Bc ? d : Bc;
	Tr = N / Br;
	Tc = N / Bc;
	
	printf("Tr: %d Br: %d Tc: %d Bc: %d\n", Tr, Br, Tc, Bc);
	
	if ((l = calloc(N, sizeof(double))) == NULL)
		return 1;
	
	if ((m = calloc(N, sizeof(double))) == NULL)
		return 1;
	
	if ((Pij = calloc(Bc, sizeof(double))) == NULL)
		return 1;
	
	for (int i = 0; i < N; i++)
		m[i] = -INFINITY;
	
	for (int j = 0; j < Tc; j++) {
		double *Kj, *Vj;
		
		Kj = K + j * Bc * d;
		Vj = V + j * Bc * d;
		
		for (int i = 0; i < Tr; i++) {
			double *Oi, *Qi, *mi, *li;
			
			Oi = O + i * Br * d;
			Qi = Q + i * Br * d;
			mi = m + i * Br;
			li = l + i * Br;
			
			for (int ii = 0; ii < Br; ii++) {
				double sum, max, li_new, mi_new, eij, eli;
				
				max = -INFINITY;

				for (int jj = 0; jj < Bc; jj++) {
					sum = 0;
					
					for (int kk = 0; kk < d; kk++) {
						double x, y;
						
						x = Qi[ii * d + kk];
						y = Kj[jj * d + kk];
						
						sum += x * y;
					}
					
					Pij[jj] = sum;
					max = sum > max ? sum : max;
				}
				
				sum = 0;

				for (int jj = 0; jj < Bc; jj++) {
					double *x = &Pij[jj];
					
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
					double x = eij * Pij[jj];

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
	double *Q, *K, *V, *O, dur;
	int N, d, M;
	
	if (argc != 4) {
		printf("Usage: N d M\n");
		return EXIT_FAILURE;
	}

	N = atoi(argv[1]);
	d = atoi(argv[2]);
	M = atoi(argv[3]);
	
	cnt = N * d;
	
	printf("Initializing data...");
	
	if ((Q = calloc(cnt, sizeof(double))) == NULL)
		goto mem_failure;
	
	if ((K = calloc(cnt, sizeof(double))) == NULL)
		goto mem_failure;
	
	if ((V = calloc(cnt, sizeof(double))) == NULL)
		goto mem_failure;

	if ((O = calloc(cnt, sizeof(double))) == NULL)
		goto mem_failure;
	
	for (size_t i = 0; i < cnt; i++) {
		Q[i] = K[i] = V[i] = 1;
	}
	
	printf("done.\n");
	printf("Warming up...");
	
	if (flash_attention(O, Q, K, V, N, d, M))
		goto attn_failure;
	
	printf("done.\n");	
	printf("Running flash_attention...");
	
	gettimeofday(&ts, NULL);
	
	if (flash_attention(O, Q, K, V, N, d, M))
		goto attn_failure;
	
	gettimeofday(&te, NULL);
	
	printf("done.\n");
	
	dur = (double)(te.tv_usec - ts.tv_usec) / 1e6 + 
		      (double)(te.tv_sec - ts.tv_sec);

	printf("L2 norm is %lf (should be %lf)\n", L2(O, cnt), sqrt(cnt));
	printf("Duration: %lf\n", dur);
	
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
