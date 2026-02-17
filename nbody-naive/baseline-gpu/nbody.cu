#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <cstdint>
#include "util.cu.h"

__host__ int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

__global__ void accelerateAllPar(const double4* pos_mass, double3*  accel, int N) {
  extern __shared__ double shmem[];
  
  D3 acc(0.0, 0.0, 0.0);
  double4 pos_i;

  { // read the position of the current body
    double4* pos = (double4*)shmem;
    if(threadIdx.x == 0) {
      pos[0] = pos_mass[blockIdx.x]; 
    }
    __syncthreads();
    pos_i = pos[0];
  }

  // per-thread partial aggregation of results
  for(int tid = threadIdx.x; tid < N; tid += blockDim.x) {
      double3 dist;
      double4 pos_j_k = pos_mass[tid];

      dist.x = pos_j_k.x - pos_i.x;
      dist.y = pos_j_k.y - pos_i.y;
      dist.z = pos_j_k.z - pos_i.z;
      double dist_sq = dist.x*dist.x + dist.y*dist.y + dist.z*dist.z;
      double inv_dist = 1.0 / sqrt(dist_sq + 1e-9);
      double inv_dist3 = inv_dist*inv_dist*inv_dist;

      acc.x += dist.x * pos_j_k.w * inv_dist3;
      acc.y += dist.y * pos_j_k.w * inv_dist3;
      acc.z += dist.z * pos_j_k.w * inv_dist3;
  }
  __syncthreads();
  
  { // publish partial results in shared memory and perform block-level scan
    volatile D3* accs = (volatile D3*)shmem;
    accs[threadIdx.x].x = acc.x;
    accs[threadIdx.x].y = acc.y;
    accs[threadIdx.x].z = acc.z;
    __syncthreads();
    
    // block-level scan
    acc = scanIncBlock<AddD3>(accs, threadIdx.x);
  }
  //__syncthreads();
  
  // last thread publishes the acceleration result to global memory
  if (threadIdx.x == blockDim.x-1) {
    double3 acc_res = make_double3(acc.x, acc.y, acc.z);
    accel[blockIdx.x] = acc_res;
  }
}


__global__ void accelerateOutPar(const double4* pos_mass, double3*  accel, int N) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int num_threads = blockDim.x;

  int start_row = bid * num_threads;
  int end_row = min(start_row + num_threads, N);
  int actual_threads = end_row - start_row;

  extern __shared__ double4 pos_j[];

  double3 acc = make_double3(0.0, 0.0, 0.0);
  double3 dist;
  double4 pos_i;

  if (tid < actual_threads) {
    pos_i = pos_mass[start_row + tid];
  }

  for (int j = 0; j < N; j += num_threads) {
    if (j + tid < N) {
      pos_j[tid] = pos_mass[j + tid];
    }
    __syncthreads();

    if (tid < actual_threads) {

      for (int k = 0; k < min(num_threads, N - j); k++) {
        double4 pos_j_k = pos_j[k];
        dist.x = pos_j_k.x - pos_i.x;
        dist.y = pos_j_k.y - pos_i.y;
        dist.z = pos_j_k.z - pos_i.z;
        double dist_sq = dist.x*dist.x + dist.y*dist.y + dist.z*dist.z;
        double inv_dist = 1.0 / sqrt(dist_sq + 1e-9);
        double inv_dist3 = inv_dist*inv_dist*inv_dist;
        acc.x += dist.x * pos_j_k.w * inv_dist3;
        acc.y += dist.y * pos_j_k.w * inv_dist3;
        acc.z += dist.z * pos_j_k.w * inv_dist3;
      }
    }

    __syncthreads();
  }

  if (tid < actual_threads) {
    accel[start_row + tid] = acc;
  }
}


__global__ void update(double4* pos, double3* vel, double3* accel, double dt, int N) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N) {
    double4 cpos = pos[i];
    double3 cvel = vel[i];
    double3 caccel = accel[i];
    double3 nvel = make_double3(cvel.x + dt * caccel.x,
                                cvel.y + dt * caccel.y,
                                cvel.z + dt * caccel.z);
    double4 npos = make_double4(cpos.x + dt * nvel.x,
                                cpos.y + dt * nvel.y,
                                cpos.z + dt * nvel.z,
                                cpos.w);
    pos[i] = npos;
    vel[i] = nvel;
  }
}


void nbody(double4* positions, double3* velocities, double dt, int n, int steps, int block_size) {
  double3* accel;
  cudaMalloc(((void**)&accel), n * sizeof(double3));

  const int num_threads = min(n, block_size);
  const int num_blocks = (n + num_threads - 1) / num_threads;
  dim3 grid(num_blocks);
  dim3 block(num_threads);
    
  int block_size_all;
  if(n <= 256) {
    block_size_all = n;
  } else if (n <= 1024) {
    block_size_all = 256;
  } else {
    block_size_all = min(1024, n/4);
  }
  dim3 blockAll(block_size_all);

  const int threshold = 16384;

  if(n > threshold) {
    for (int i = 0; i < steps; ++i) {
      accelerateOutPar<<<grid, block, num_threads * sizeof(double4)>>>(positions, accel, n);
      update<<<grid, block>>>(positions, velocities, accel, dt, n);
    }
  } else { // n <= 1024
    for (int i = 0; i < steps; ++i) {
      accelerateAllPar<<<n, blockAll, block_size_all * sizeof(D3)>>>(positions, accel, n);
      update<<<grid, block>>>(positions, velocities, accel, dt, n);
    }
  }

  cudaDeviceSynchronize();  
  cudaFree(accel);
}


void init(double4* positions, double3* velocities, int n)
{
  for (int i = 0; i < n; i++) {
    positions[i].x = i;
    positions[i].y = 2.0 * i;
    positions[i].z = 3.0 * i;
    positions[i].w = 1.0;  // mass
    velocities[i].x = 0.0;
    velocities[i].y = 0.0;
    velocities[i].z = 0.0;
  }
}

int main(int argc, char **argv)
{
  if (argc != 4) {
    printf("Usage: N STEPS RUNS\n");
    return EXIT_FAILURE;
  }

  int n = atoi(argv[1]);
  int steps = atoi(argv[2]);
  int runs = atoi(argv[3]);

  int block_size = 512;

  fprintf(stderr,
          "CUDA nbody with %d bodies, %d steps, %d runs.\n",
          n, steps, runs);
  fprintf(stderr, "Block size %d\n", block_size);

  cudaSetDevice(0);

  double4* positions = (double4*)malloc(n * sizeof(double4));  // includes masses
  double3* velocities = (double3*)malloc(n * sizeof(double3));
  double4* pos_dev;
  double3* vel_dev;
  cudaMalloc((void**)&pos_dev, n * sizeof(double4));
  cudaMalloc((void**)&vel_dev, n * sizeof(double3));

  init(positions, velocities, n);

  double* runtimes = (double*)calloc(runs,sizeof(double));

  // Intentional - we discard first run.
  for (int r = 0; r <= runs; r++) {
    fprintf(stderr, "Run %d\n", r);
    cudaMemcpy(pos_dev, positions, n * sizeof(double4), cudaMemcpyHostToDevice);
    cudaMemcpy(vel_dev, velocities, n * sizeof(double3), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    nbody(pos_dev, vel_dev, 0.01, n, steps, block_size);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );
    gettimeofday(&tv2, NULL);

    double duration =
      (double) (tv2.tv_usec - tv1.tv_usec) / 1e6 +
      (double) (tv2.tv_sec - tv1.tv_sec);

    if (r != 0) {
      runtimes[r-1] = duration;
    }
  }

  for (int r = 0; r < runs; r++) {
      double gflops = 1e-9 * (19 * n * n + 12 * n) * steps;
    printf("Baseline (GPU),n=%d,%f\n", n, gflops / (double)runtimes[r]);
  }

  cudaFree(pos_dev);
  cudaFree(vel_dev);

  free(positions);
  free(velocities);
  free(runtimes);

  return EXIT_SUCCESS;
}
