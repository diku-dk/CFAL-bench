/***********************************************************
 * A Cuda translation of the                               *
 *   faithful implementation of Algorithm 1 from the paper *
 *   proposed by Aaron W. Hsu <arcfide@sacrideo.us>        *
 * Date: May 2024                                          *
 ***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define GPU_RUNS 100

__host__ int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}


__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__global__ void alg1Ker ( int d, int N, int Tc
                        , float* Q, float* K, float* V
                        , float* O, float* ms, float* ls
) {
  // shared memory size: (Bc*(d+1) + 2*Br*d + Br*Bc + 6*Br) * sizeof(float)
  extern __shared__ char sh_mem_char[];

  const int Br = blockDim.y;
  const int Bc = blockDim.x;

  float* Oi   = (float*)sh_mem_char;  // [Br][d]
  float* Qi   = Oi   + Br*d;          // [Br][d]
  float* Pij  = Qi   + Br*d;          // [Br][Bc]
  
  float* Kj   = Pij  + Br*Bc;         // [Bc][d+1]
  float* Vj   = Kj;                   // [Bc][d]

  float* maxs = Kj  + Bc*(d+1);       // [Br]

  float* sums = maxs + Br;            // [Br]
  float* es   = sums + Br;            // [Br]
  float* el   = es   + Br;            // [Br]
  float* li   = el   + Br;            // [Br]
  float* mi   = li   + Br;            // [Br]

  int i  = blockIdx.x;
  int ii = threadIdx.y; // ii < Br
  int jj = threadIdx.x; // jj < Bc

  const int tid = ii*Bc + jj;

  // initialize mi, li
  for (int t = tid; t < Br; t+=Br*Bc) {
    mi[t]   = -INFINITY;
    li[t]   = 0;
  }

  // copy Qi from global to shared memory
  // can be optimized a bit by normalizing the loop
  for (int t = tid; t < Br*d; t+=Br*Bc) {
    int64_t glb_ind = i * Br * d + t;
    Qi[t] = Q[glb_ind];
    Oi[t] = 0;
  }

  __syncthreads();

  for (int j = 0; j < Tc; j++) {

    // initialize maxs, sums
    for (int t = tid; t < Br; t+=Br*Bc) {
      sums[t] = 0;
      maxs[t] = -INFINITY;
    }

    // copy Kj from global to shared memory;
    // can be optimized a bit by normalizing the loop
    // Kj is padded to avoid very expensive bank conflicts in mmm.
    for (int t = tid; t < Bc*d; t+=Br*Bc) {
      int64_t glb_ind = j * Bc * d + t;
      int q = t / d;
      int r = t - q*d; 
      Kj[q*(d+1) + r] = K[glb_ind];
      //Kj[t] = K[glb_ind];
    }
    __syncthreads();

    ////////////////////////////////////
    // first matrix multiplication
    ////////////////////////////////////
    float pij = 0.0;
    {
      for (int kk = 0; kk < d; kk++) {
        pij +=
          Qi[ii * d + kk] *
          Kj[jj*(d+1) + kk]; //Kj[jj * d + kk] ;
      }
    }

    ////////////////////////
    // reductions
    ////////////////////////
    atomicMaxFloat(&maxs[ii], pij);
    __syncthreads();

    {
      pij = exp(pij - maxs[ii]);
      //Pij[ii*Bc + jj] = pij;
      atomicAdd(&sums[ii], pij);
    }
    __syncthreads();

    if(tid < Br) {
      const int ii = tid;
      float mi_old = mi[ii];
      float mx = maxs[ii];
      float mi_new = (mi_old > mx) ? mi_old : mx;
      float eij = exp(mx - mi_new);
      float eli = li[ii] * exp(mi_old - mi_new);
      float li_new = eli + sums[ii] * eij;

      mi[ii] = mi_new;
      li[ii] = li_new;
      es[ii] = eij;
      el[ii] = eli;
    }
    __syncthreads();

    Pij[ii*Bc+jj] = es[ii]*pij;

    // copy Vj from global to shared memory
    for (int t = tid; t < Bc*d; t+=Br*Bc) {
      int64_t glb_ind = j * Bc * d + t;
      Vj[t] = V[glb_ind];
    }
    __syncthreads();

    ////////////////////////////////////
    // second matrix multiplication
    ////////////////////////////////////
    for(int k = 0; k < d / Bc; k++) {
      int kk = k * Bc + jj;
      int ik = ii * d + kk;
      float oi_ik = Oi[ik] * el[ii];

      for(int jjj = 0; jjj < Bc; jjj++) {
        int jk = jjj * d + kk;
        float x = Pij[ii*Bc + jjj];
        float y = Vj[jk];
        oi_ik += x * y;
      }

      Oi[ik] = oi_ik / li[ii];
    }
    __syncthreads();
  }
  
  // copy Oi back to global memory
  for (int t = tid; t < Br*d; t+=Br*Bc) {
    int64_t glb_ind = i * Br * d + t;
    O[glb_ind] = Oi[t];
  }

  // copy ms, ls back to global memory
  for (int t = tid; t < Br; t+=Br*Bc) {
    int64_t glb_ind = i * Br + t;
    ms[glb_ind] = mi[t];
    ls[glb_ind] = li[t];
  }
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

__host__ float
L2(float *x, size_t c)
{
        float sum;

        sum = 0;

        for (size_t i = 0; i < c; i++)
                sum += x[i] * x[i];

        return sqrt(sum);
}

__host__ int
flash_attention(float* m_d, float* l_d, float *O_d, float *Q_d, float *K_d, float *V_d, int N, int d, int M)
{
    int Br, Bc, Tr, Tc;

    Bc = M / (4 * d);
    Br = d < Bc ? d : Bc;
    Tr = N / Br;
    Tc = N / Bc;

    // setup execution parameters
    dim3 block(Bc, Br, 1);
    dim3 grid (Tr,  1, 1);
    const size_t shmem_size = (Bc*(d+1) + 2*Br*d + Br*Bc + 6*Br) * sizeof(float);
    //printf("\nShared memory size: %d, Bc=%d, Br=%d, Tc: %d, Tr: %d, d: %d, N: %d\n", shmem_size, Bc, Br, Tc, Tr, d, N);

    alg1Ker<<<grid, block, shmem_size>>>( d, N, Tc, Q_d, K_d, V_d, O_d, m_d, l_d );
    //cudaDeviceSynchronize();

#if 0
    {
        float* m_h = (float*) malloc(N*sizeof(float));
        cudaMemcpy(m_h, m_d, N*sizeof(float), cudaMemcpyDeviceToHost);
        {
            printf("(N,d,Br,Bc,Tr,Tc)=(%d,%d,%d,%d,%d,%d), ms:\n    ", N,d,Br,Bc,Tr,Tc);
            for(int q=0; q<Br; q++) {
                printf(", %f", m_h[q]);
            }
            printf("\n");
        }

        free(m_h);
    }
#endif

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

    if ((Q = (float*)calloc(cnt, sizeof(float))) == NULL)
        goto mem_failure;

    if ((K = (float*)calloc(cnt, sizeof(float))) == NULL)
        goto mem_failure;

    if ((V = (float*)calloc(cnt, sizeof(float))) == NULL)
        goto mem_failure;

    if ((O = (float*)calloc(cnt, sizeof(float))) == NULL)
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

    {
        float *Q_d, *K_d, *V_d, *O_d, *m_d, *l_d;

        cudaSetDevice(1);

        // allocate memory on device
        cudaMalloc((void**) &Q_d, cnt*sizeof(float));
        cudaMalloc((void**) &O_d, cnt*sizeof(float));
        cudaMalloc((void**) &V_d, cnt*sizeof(float));
        cudaMalloc((void**) &K_d, cnt*sizeof(float));
        cudaMalloc((void**) &m_d, N*sizeof(float));
        cudaMalloc((void**) &l_d, N*sizeof(float));

        // copy host memory to device
        cudaMemcpy(Q_d, Q, cnt*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(V_d, V, cnt*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(K_d, K, cnt*sizeof(float), cudaMemcpyHostToDevice);

        fprintf(stderr, "Warming up...");

        flash_attention(m_d, l_d, O_d, Q_d, K_d, V_d, N, d, M);
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );

        fprintf(stderr, "done.\n");
        fprintf(stderr, "Running flash_attention...");

        cudaFuncSetAttribute(alg1Ker, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

        gettimeofday(&ts, NULL);

        for(int i=0; i<GPU_RUNS; i++) {
            flash_attention(m_d, l_d, O_d, Q_d, K_d, V_d, N, d, M);
        }
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );

        gettimeofday(&te, NULL);

        fprintf(stderr, "done.\n");

        dur = (double)(te.tv_usec - ts.tv_usec) / 1e6 +
                          (double)(te.tv_sec - ts.tv_sec);
        dur = dur / GPU_RUNS;

        cudaMemcpy(O, O_d, cnt*sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

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
                "Compute rate: %lf Gflops/s, runtime: %lf\n",
                (4.0 * d + 5.0) * N * N / dur / 1e9,  dur*1e6);

        cudaFree(Q_d);
        cudaFree(K_d);
        cudaFree(V_d);
        cudaFree(O_d);
        cudaFree(m_d);
        cudaFree(l_d);
    }

    free(Q);
    free(K);
    free(V);
    free(O);

    return EXIT_SUCCESS;

mem_failure:
        printf("Failed to allocate memory.\n");
        return EXIT_FAILURE;
}
