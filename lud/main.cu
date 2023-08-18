/**
 * This LUD Cuda implementation is essentially
 * taken from Rodinia_3.1 suite and hopefully we will get
 * to slightly modify in the future, e.g., by using block
 * and register tiling for the batched matrix-matrix-like
 * multiplication.
 */

#include "kernels.cu.h"
#include "helper.h"
#include "goldenSeq.h"

#define GPU_RUNS   100
#define BLOCK      32
#define ERR        0.01

typedef enum _FUNC_RETURN_CODE {
    RET_SUCCESS,
    RET_FAILURE
}func_ret_t;


// Generate well-conditioned matrix internally  by Ke Wang 2013/08/07 22:20:06
template<class ElTp>
func_ret_t
create_matrix(ElTp **mp, int size){
  ElTp *m;
  int i,j;
  ElTp lamda = -0.001;
  ElTp coe[2*size-1];
  ElTp coe_i = 0.0;

  for(i=0; i < size; i++) {
    coe_i = 10*exp(lamda*i); 
    j=size-1+i;     
    coe[j]=coe_i;
    j=size-1-i;     
    coe[j]=coe_i;
  }

  m = (ElTp*) malloc(sizeof(ElTp)*size*size);
  if ( m == NULL ) {
      return RET_FAILURE;
  }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
        m[i*size+j]=coe[size-1-i+j];
      }
  }

  *mp = m;

  return RET_SUCCESS;
}

template<class ElTp, int B>
void runAll( const int N ) {
  func_ret_t ret;
  ElTp *m, *d_m, *mm, *d_mm;

  size_t size = N * N;
  size_t mem_size = size * sizeof(ElTp);

  // create matrix
  ret = create_matrix<float>(&m, N);
  if (ret != RET_SUCCESS) {
    m = NULL;
    printf("Error create matrix internally size=%d\n", N);
    exit(EXIT_FAILURE);
  }

  mm = (ElTp*) malloc(mem_size);
  if(mm==NULL) {
    printf("Alocation of mm: insuficcient memory!\n");
    exit(1);
  }
  memcpy(mm, m, mem_size);

  // allocate and init device
  cudaMalloc((void**)&d_m,  mem_size);
  cudaMalloc((void**)&d_mm, mem_size);

  cudaMemcpy( d_m, m, mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mm, m, mem_size, cudaMemcpyHostToDevice);

  // golden sequential
  lud_base<ElTp>(m, N); 

  
  { // cuda implementation:
    // dry run
    lud_cuda<ElTp, B>(d_mm, N);
    cudaMemcpy(d_mm, d_m, mem_size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 

    for(int i=0; i<GPU_RUNS; i++) {
        cudaMemcpy(d_mm, d_m, mem_size, cudaMemcpyDeviceToDevice);
        lud_cuda<ElTp, B>(d_mm, N);
    }
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;
    gpuAssert( cudaPeekAtLastError() );

    double flops = 0;
    for(int i=0; i<N; i++) {
        flops += (2*i*N - 2*i*i) + (N-i-1)*(2*i+1);
    }

    double gigaFlopsPerSec = (flops * 1.0e-3f) / elapsed; 
    printf("Cuda implementation of LUD of size (%d x %d) runs in %lu microsecs, GFlops/sec: %.2f\n"
          , N, N, elapsed, gigaFlopsPerSec);

    // copy back to host (from device)
    cudaMemcpy(mm, d_mm, mem_size, cudaMemcpyDeviceToHost);
  }

  // validation
  
  validate<ElTp>(m, mm, size, ERR);

  // free memory
  free(m);
  free(mm);
  cudaFree(d_m);
  cudaFree(d_mm);
}

int
main ( int argc, char *argv[] ) {
  int N;
  printf("Block size (statically determined) = %d\n", BLOCK);

  // 1. Reading dimensions;
  //    assumes an N x N matrix
  if (argc == 2) {
      N = atoi(argv[1]);
  } else {
      printf("Usage: %s <num_rows>\n", argv[0]);
      exit(1);
  }
    
  if( (N < 2) || ((N % BLOCK) != 0) ) {
    printf("The dimension values must be a multiple of BLOCK_SIZE\n");
    exit(1);
  }

  runAll<float, BLOCK>(N);
  return 0;
}
