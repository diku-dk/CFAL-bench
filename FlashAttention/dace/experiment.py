import cupy as cp
import cupyx as cpx
import numpy as np
import math
import time


kernel_code = '''
extern "C" __global__ void __launch_bounds__(({BM} * {BN}) / ({TM} * {TN}), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {{
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  const int totalResultsBlocktile = {BM} * {BN};
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const int numThreadsBlocktile = totalResultsBlocktile / ({TM} * {TN});

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % ({BN} / {TN});
  const int threadRow = threadIdx.x / ({BN} / {TN});

  // allocate space for the current blocktile in smem
  __shared__ float As[{BM} * {BK}];
  __shared__ float Bs[{BK} * {BN}];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * {BM} * K;
  B += cCol * {BN};
  C += cRow * {BM} * N + cCol * {BN};

  // calculating the indices that this thread will load into SMEM
  const int innerRowA = threadIdx.x / {BK};
  const int innerColA = threadIdx.x % {BK};
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const int strideA = numThreadsBlocktile / {BK};
  const int innerRowB = threadIdx.x / {BN};
  const int innerColB = threadIdx.x % {BN};
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const int strideB = numThreadsBlocktile / {BN};

  // allocate thread-local cache for results in registerfile
  float threadResults[{TM} * {TN}] = {{0.0}};
  // register caches for As and Bs
  float regM[{TM}] = {{0.0}};
  float regN[{TN}] = {{0.0}};

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += {BK}) {{
    // populate the SMEM caches
    for (int loadOffset = 0; loadOffset < {BM}; loadOffset += strideA) {{
      As[(innerRowA + loadOffset) * {BK} + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }}
    for (int loadOffset = 0; loadOffset < {BK}; loadOffset += strideB) {{
      Bs[(innerRowB + loadOffset) * {BN} + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }}
    __syncthreads();

    // advance blocktile
    A += {BK};     // move BK columns to right
    B += {BK} * N; // move BK rows down

    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < {BK}; ++dotIdx) {{
      // block into registers
      for (int i = 0; i < {TM}; ++i) {{
        regM[i] = As[(threadRow * {TM} + i) * {BK} + dotIdx];
      }}
      for (int i = 0; i < {TN}; ++i) {{
        regN[i] = Bs[dotIdx * {BN} + threadCol * {TN} + i];
      }}
      for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
        for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
          threadResults[resIdxM * {TN} + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }}
      }}
    }}
    __syncthreads();
  }}

  // write out the results
  for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
    for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
      C[(threadRow * {TM} + resIdxM) * N + threadCol * {TN} + resIdxN] =
          alpha * threadResults[resIdxM * {TN} + resIdxN] +
          beta * C[(threadRow * {TM} + resIdxM) * N + threadCol * {TN} + resIdxN];
    }}
  }}
}}
'''


kernel_code_2 = '''
extern "C" __global__ void __launch_bounds__(({BM} * {BN}) / ({TM} * {TN}), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *origA,
                       const float *origB, float beta, float *origC) {{
  // const int cRow = blockIdx.y;
  // const int cCol = blockIdx.x;

  const int totalResultsBlocktile = {BM} * {BN};
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const int numThreadsBlocktile = totalResultsBlocktile / ({TM} * {TN});

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % ({BN} / {TN});
  const int threadRow = threadIdx.x / ({BN} / {TN});

  // allocate space for the current blocktile in smem
  __shared__ float As[{BM} * {BK}];
  __shared__ float Bs[{BK} * {BN}];

  const float *A;
  const float *B;
  float *C;

  int blocksM = M / {BM};
  int blocksN = N / {BN};

  for (int cCol = 0; cCol < blocksN; ++cCol) {{
    for (int cRow = 0; cRow < blocksM; ++cRow) {{
  
  A = origA;
  B = origB;
  C = origC;

  // Move blocktile to beginning of A's row and B's column
  A += cRow * {BM} * K;
  B += cCol * {BN};
  C += cRow * {BM} * N + cCol * {BN};

  // if (threadIdx.x == 0) {{
  //   printf("cRow: %d, cCol: %d\\n", cRow, cCol);
  // }}

  // calculating the indices that this thread will load into SMEM
  const int innerRowA = threadIdx.x / {BK};
  const int innerColA = threadIdx.x % {BK};
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const int strideA = numThreadsBlocktile / {BK};
  const int innerRowB = threadIdx.x / {BN};
  const int innerColB = threadIdx.x % {BN};
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const int strideB = numThreadsBlocktile / {BN};

  // allocate thread-local cache for results in registerfile
  float threadResults[{TM} * {TN}] = {{0.0}};
  // register caches for As and Bs
  float regM[{TM}] = {{0.0}};
  float regN[{TN}] = {{0.0}};

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += {BK}) {{
    // populate the SMEM caches
    for (int loadOffset = 0; loadOffset < {BM}; loadOffset += strideA) {{
      As[(innerRowA + loadOffset) * {BK} + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }}
    for (int loadOffset = 0; loadOffset < {BK}; loadOffset += strideB) {{
      Bs[(innerRowB + loadOffset) * {BN} + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }}
    __syncthreads();

    // advance blocktile
    A += {BK};     // move BK columns to right
    B += {BK} * N; // move BK rows down

    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < {BK}; ++dotIdx) {{
      // block into registers
      for (int i = 0; i < {TM}; ++i) {{
        regM[i] = As[(threadRow * {TM} + i) * {BK} + dotIdx];
      }}
      for (int i = 0; i < {TN}; ++i) {{
        regN[i] = Bs[dotIdx * {BN} + threadCol * {TN} + i];
      }}
      for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
        for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
          threadResults[resIdxM * {TN} + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }}
      }}
    }}
    __syncthreads();
  }}

  // write out the results
  for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
    for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
      C[(threadRow * {TM} + resIdxM) * N + threadCol * {TN} + resIdxN] =
          alpha * threadResults[resIdxM * {TN} + resIdxN] +
          beta * C[(threadRow * {TM} + resIdxM) * N + threadCol * {TN} + resIdxN];
    }}
  }}

  __syncthreads();

  }} // cCol
  }} // cRow

}}
'''


kernel_code_3 = '''
extern "C" __global__ void __launch_bounds__(({BM} * {BN}) / ({TM} * {TN}), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *origA,
                       const float *origB, float beta, float *origC) {{
  // const int cRow = blockIdx.y;
  // const int cCol = blockIdx.x;

  const int totalResultsBlocktile = {BM} * {BN};
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const int numThreadsBlocktile = totalResultsBlocktile / ({TM} * {TN});

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % ({BN} / {TN});
  const int threadRow = threadIdx.x / ({BN} / {TN});

  // calculating the indices that this thread will load into SMEM
  const int innerRowA = threadIdx.x / {BK};
  const int innerColA = threadIdx.x % {BK};
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const int strideA = numThreadsBlocktile / {BK};
  const int innerRowB = threadIdx.x / {BN};
  const int innerColB = threadIdx.x % {BN};
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const int strideB = numThreadsBlocktile / {BN};
  const int innerRowC = threadIdx.x / {BN};
  const int innerColC = threadIdx.x % {BN};
  const int strideC = numThreadsBlocktile / {BN};

  // allocate space for the current blocktile in smem
  __shared__ float As[{BM} * {BK}];
  __shared__ float Bs[{BK} * {BN}];
  // __shared__ float Cs[{BM} * {BN}];
  // __shared__ float Ds[{BN} * {BL}];

  const float *A;
  const float *B;
  float *C;
  // const float *D;

  int blocksM = M / {BM};
  int blocksN = N / {BN};
  // int blocksL = K / {BL};

  for (int cCol = 0; cCol < blocksN; ++cCol) {{
  
  B = origB;
  B += cCol * {BN};

  for (int bkIdx = 0; bkIdx < K; bkIdx += {BK}) {{
    // populate the SMEM caches
    for (int loadOffset = 0; loadOffset < {BK}; loadOffset += strideB) {{
      Bs[(innerRowB + loadOffset) * {BN} + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }}

    // advance blocktile
    B += {BK} * N; // move BK rows down
  }}

    for (int cRow = 0; cRow < blocksM; ++cRow) {{
  
  A = origA;
  B = origB;
  C = origC;
  // D = origD;

  // Move blocktile to beginning of A's row and B's column
  A += cRow * {BM} * K;
  B += cCol * {BN};
  C += cRow * {BM} * N + cCol * {BN};
  // D += cCol * {BN} * K;

  // if (threadIdx.x == 0) {{
  //   printf("cRow: %d, cCol: %d\\n", cRow, cCol);
  // }}

  // allocate thread-local cache for results in registerfile
  float threadResults[{TM} * {TN}] = {{0.0}};
  // register caches for As and Bs
  float regM[{TM}] = {{0.0}};
  float regN[{TN}] = {{0.0}};

  // for (int loadOffset = 0; loadOffset < {BM}; loadOffset += strideC) {{
  //   Cs[(innerRowC + loadOffset) * {BN} + innerColC] =
  //       C[(innerRowC + loadOffset) * N + innerColC];
  // }}

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += {BK}) {{
    // populate the SMEM caches
    for (int loadOffset = 0; loadOffset < {BM}; loadOffset += strideA) {{
      As[(innerRowA + loadOffset) * {BK} + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }}
    // for (int loadOffset = 0; loadOffset < {BK}; loadOffset += strideB) {{
    //   Bs[(innerRowB + loadOffset) * {BN} + innerColB] =
    //       B[(innerRowB + loadOffset) * N + innerColB];
    // }}
    __syncthreads();

    // advance blocktile
    A += {BK};     // move BK columns to right
    B += {BK} * N; // move BK rows down

    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < {BK}; ++dotIdx) {{
      // block into registers
      for (int i = 0; i < {TM}; ++i) {{
        regM[i] = As[(threadRow * {TM} + i) * {BK} + dotIdx];
      }}
      for (int i = 0; i < {TN}; ++i) {{
        regN[i] = Bs[dotIdx * {BN} + threadCol * {TN} + i];
      }}
      for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
        for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
          threadResults[resIdxM * {TN} + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }}
      }}
    }}
    __syncthreads();
  }}

  // write out the results
  for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
    for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
      C[(threadRow * {TM} + resIdxM) * N + threadCol * {TN} + resIdxN] =
          alpha * threadResults[resIdxM * {TN} + resIdxN] +
          beta * C[(threadRow * {TM} + resIdxM) * N + threadCol * {TN} + resIdxN];
    }}
  }}

  __syncthreads();

  }} // cCol
  }} // cRow

}}
'''


kernel_code_4 = '''
extern "C" __global__ void __launch_bounds__(({BM} * {BN}) / ({TM} * {TN}), 1)
    sgemm2DBlocktiling(int M, int N, int K, const float *origA, const float *origB, float *origC) {{
  // const int cRow = blockIdx.y;
  // const int cCol = blockIdx.x;
  const int cRow = blockIdx.x;

  const int totalResultsBlocktile = {BM} * {BN};
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const int numThreadsBlocktile = totalResultsBlocktile / ({TM} * {TN});

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % ({BN} / {TN});
  const int threadRow = threadIdx.x / ({BN} / {TN});

  // allocate space for the current blocktile in smem
  __shared__ float As[{BM} * {BK}];
  __shared__ float Bs[{BK} * {BN}];

  const float *A;
  const float *B;
  float *C;

  int blocksM = M / {BM};
  int blocksN = N / {BN};

  for (int cCol = 0; cCol < blocksN; ++cCol) {{
  
  A = origA;
  B = origB;
  C = origC;

  // Move blocktile to beginning of A's row and B's column
  A += cRow * {BM} * K;
  B += cCol * {BN};
  C += cRow * {BM} * N + cCol * {BN};

  // if (threadIdx.x == 0) {{
  //   printf("cRow: %d, cCol: %d\\n", cRow, cCol);
  // }}

  // calculating the indices that this thread will load into SMEM
  const int innerRowA = threadIdx.x / {BK};
  const int innerColA = threadIdx.x % {BK};
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const int strideA = numThreadsBlocktile / {BK};
  const int innerRowB = threadIdx.x / {BN};
  const int innerColB = threadIdx.x % {BN};
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const int strideB = numThreadsBlocktile / {BN};

  // allocate thread-local cache for results in registerfile
  float threadResults[{TM} * {TN}] = {{0.0}};
  // register caches for As and Bs
  float regM[{TM}] = {{0.0}};
  float regN[{TN}] = {{0.0}};

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += {BK}) {{
    // populate the SMEM caches
    for (int loadOffset = 0; loadOffset < {BM}; loadOffset += strideA) {{
      As[(innerRowA + loadOffset) * {BK} + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }}
    for (int loadOffset = 0; loadOffset < {BK}; loadOffset += strideB) {{
      Bs[(innerRowB + loadOffset) * {BN} + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }}
    __syncthreads();

    // advance blocktile
    A += {BK};     // move BK columns to right
    B += {BK} * N; // move BK rows down

    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < {BK}; ++dotIdx) {{
      // block into registers
      for (int i = 0; i < {TM}; ++i) {{
        regM[i] = As[(threadRow * {TM} + i) * {BK} + dotIdx];
      }}
      for (int i = 0; i < {TN}; ++i) {{
        regN[i] = Bs[dotIdx * {BN} + threadCol * {TN} + i];
      }}
      for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
        for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
          threadResults[resIdxM * {TN} + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }}
      }}
    }}
    __syncthreads();
  }}

  // write out the results
  for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
    for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
      C[(threadRow * {TM} + resIdxM) * N + threadCol * {TN} + resIdxN] =
          threadResults[resIdxM * {TN} + resIdxN];
    }}
  }}

  __syncthreads();

  }} // cCol

}}
'''


kernel_code_5 = '''
extern "C" __global__ void __launch_bounds__(({BM} * {BN}) / ({TM} * {TN}), 1)
    sgemm2DBlocktiling(int M, int N, int K, const float *origA, const float *origB, float *origC) {{
  // const int cRow = blockIdx.y;
  // const int cCol = blockIdx.x;
  const int cRow = blockIdx.x;

  const int totalResultsBlocktile = {BM} * {BN};
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const int numThreadsBlocktile = totalResultsBlocktile / ({TM} * {TN});

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % ({BN} / {TN});
  const int threadRow = threadIdx.x / ({BN} / {TN});

  // allocate space for the current blocktile in smem
  __shared__ float As[{BM} * {BK}];
  __shared__ float Bs[{BK} * {BN}];
  __shared__ float Cs[{BM} * {BN}];

  const float *A;
  const float *B;
  float *C;

  int blocksM = M / {BM};
  int blocksN = N / {BN};

  for (int cCol = 0; cCol < blocksN; ++cCol) {{
  
  A = origA;
  B = origB;
  C = origC;

  // Move blocktile to beginning of A's row and B's column
  A += cRow * {BM} * K;
  B += cCol * {BN};
  C += cRow * {BM} * N + cCol * {BN};

  // if (threadIdx.x == 0) {{
  //   printf("cRow: %d, cCol: %d\\n", cRow, cCol);
  // }}

  // calculating the indices that this thread will load into SMEM
  const int innerRowA = threadIdx.x / {BK};
  const int innerColA = threadIdx.x % {BK};
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const int strideA = numThreadsBlocktile / {BK};
  const int innerRowB = threadIdx.x / {BN};
  const int innerColB = threadIdx.x % {BN};
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const int strideB = numThreadsBlocktile / {BN};

  // allocate thread-local cache for results in registerfile
  float threadResults[{TM} * {TN}] = {{0.0}};
  // register caches for As and Bs
  float regM[{TM}] = {{0.0}};
  float regN[{TN}] = {{0.0}};

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += {BK}) {{
    // populate the SMEM caches
    for (int loadOffset = 0; loadOffset < {BM}; loadOffset += strideA) {{
      As[(innerRowA + loadOffset) * {BK} + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }}
    for (int loadOffset = 0; loadOffset < {BK}; loadOffset += strideB) {{
      Bs[(innerRowB + loadOffset) * {BN} + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }}
    __syncthreads();

    // advance blocktile
    A += {BK};     // move BK columns to right
    B += {BK} * N; // move BK rows down

    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < {BK}; ++dotIdx) {{
      // block into registers
      for (int i = 0; i < {TM}; ++i) {{
        regM[i] = As[(threadRow * {TM} + i) * {BK} + dotIdx];
      }}
      for (int i = 0; i < {TN}; ++i) {{
        regN[i] = Bs[dotIdx * {BN} + threadCol * {TN} + i];
      }}
      for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
        for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
          threadResults[resIdxM * {TN} + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }}
      }}
    }}
    __syncthreads();
  }}

  // write out to SMEM
  for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
    for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
      Cs[(threadRow * {TM} + resIdxM) * {BN} + threadCol * {TN} + resIdxN] =
          threadResults[resIdxM * {TN} + resIdxN];
    }}
  }}

  // write out the results
  for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
    for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
      C[(threadRow * {TM} + resIdxM) * N + threadCol * {TN} + resIdxN] =
          // threadResults[resIdxM * {TN} + resIdxN];
          Cs[(threadRow * {TM} + resIdxM) * {BN} + threadCol * {TN} + resIdxN];
    }}
  }}

  __syncthreads();

  }} // cCol

}}
'''


kernel_code_6 = '''
extern "C" __global__ void __launch_bounds__(({BM} * {BN}) / ({TM} * {TN}), 1)
    sgemm2DBlocktiling(int M, int N, int K,
                       const float *origA, const float *origB, float *origC, const float *origD) {{
  // const int cRow = blockIdx.y;
  // const int cCol = blockIdx.x;
  const int cRow = blockIdx.x;

  const int totalResultsBlocktile = {BM} * {BN};
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const int numThreadsBlocktile = totalResultsBlocktile / ({TM} * {TN});

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % ({BN} / {TN});
  const int threadRow = threadIdx.x / ({BN} / {TN});

  // allocate space for the current blocktile in smem
  __shared__ float As[{BM} * {BK}];
  __shared__ float Bs[{BK} * {BN}];
  // __shared__ float Cs[{BM} * {BN}];
  float *Cs = As;
  float *Ds = Bs;  // BN x BK
  // float *Os = As;  // BM x BK
  // __shared__ float Os[{BM} * {BK} * 4];
  __shared__ float Os[{BM} * {BK}];

  // for (int i = threadIdx.x; i < {BM} * {BK} * 4; i += {BM} * {BK}) {{
  //   Os[i] = 0.0;
  // }}

  const float *A;
  const float *B;
  float *C;
  const float *D;
  // float *curOs;

  int blocksM = M / {BM};
  int blocksN = N / {BN};

  for (int cCol = 0; cCol < blocksN; ++cCol) {{
  
  A = origA;
  B = origB;
  C = origC;
  D = origD;

  // Move blocktile to beginning of A's row and B's column
  A += cRow * {BM} * K;
  B += cCol * {BN};
  C += cRow * {BM} * K;
  D += cCol * {BN} * K;  // Is this correct?

  // calculating the indices that this thread will load into SMEM
  const int innerRowA = threadIdx.x / {BK};
  const int innerColA = threadIdx.x % {BK};
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const int strideA = numThreadsBlocktile / {BK};
  const int innerRowB = threadIdx.x / {BN};
  const int innerColB = threadIdx.x % {BN};
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const int strideB = numThreadsBlocktile / {BN};
  const int innerRowD = threadIdx.x / {BK};
  const int innerColD = threadIdx.x % {BK};
  const int strideD = numThreadsBlocktile / {BK};

  // allocate thread-local cache for results in registerfile
  float threadResults[{TM} * {TN}] = {{0.0}};
  // register caches for As and Bs
  float regM[{TM}] = {{0.0}};
  float regN[{TN}] = {{0.0}};

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += {BK}) {{
    // populate the SMEM caches
    for (int loadOffset = 0; loadOffset < {BM}; loadOffset += strideA) {{
      As[(innerRowA + loadOffset) * {BK} + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }}
    for (int loadOffset = 0; loadOffset < {BK}; loadOffset += strideB) {{
      Bs[(innerRowB + loadOffset) * {BN} + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }}
    __syncthreads();

    // advance blocktile
    A += {BK};     // move BK columns to right
    B += {BK} * N; // move BK rows down

    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < {BK}; ++dotIdx) {{
      // block into registers
      for (int i = 0; i < {TM}; ++i) {{
        regM[i] = As[(threadRow * {TM} + i) * {BK} + dotIdx];
      }}
      for (int i = 0; i < {TN}; ++i) {{
        regN[i] = Bs[dotIdx * {BN} + threadCol * {TN} + i];
      }}
      for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
        for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
          threadResults[resIdxM * {TN} + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }}
      }}
    }}
    __syncthreads();
  }}

  // write out to SMEM
  for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
    for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
      Cs[(threadRow * {TM} + resIdxM) * {BN} + threadCol * {TN} + resIdxN] =
          threadResults[resIdxM * {TN} + resIdxN];
    }}
  }}

  // for (int i = 0; i < {TM} * {TN}; ++i) {{
  //   threadResults[i] = 0.0;
  // }}

  // curOs = Os;

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += {BK}) {{
    // populate the SMEM caches
    for (int loadOffset = 0; loadOffset < {BK}; loadOffset += strideD) {{
      Ds[(innerRowD + loadOffset) * {BK} + innerColD] =
          D[(innerRowD + loadOffset) * K + innerColD];
    }}
    for (int loadOffset = 0; loadOffset < {BM}; loadOffset += strideD) {{
      Os[(innerRowD + loadOffset) * {BK} + innerColD] =
          C[(innerRowD + loadOffset) * K + innerColD];
    }}
    __syncthreads();

    // advance blocktile
    D += {BK};     // move BK columns to right

    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < {BN}; ++dotIdx) {{
      // block into registers
      for (int i = 0; i < {TM}; ++i) {{
        regM[i] = Cs[(threadRow * {TM} + i) * {BN} + dotIdx];
      }}
      for (int i = 0; i < {TN}; ++i) {{
        regN[i] = Ds[dotIdx * {BK} + threadCol * {TN} + i];
      }}
      for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
        for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
          // threadResults[resIdxM * {TN} + resIdxN] = regM[resIdxM] * regN[resIdxN];
          Os[(threadRow * {TM} + resIdxM) * {BK} + threadCol * {TN} + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }}
      }}
    }}

    for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
      for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
        // threadResults[resIdxM * {TN} + resIdxN] = regM[resIdxM] * regN[resIdxN];
        C[(threadRow * {TM} + resIdxM) * K + threadCol * {TN} + resIdxN] =
            Os[(threadRow * {TM} + resIdxM) * {BK} + threadCol * {TN} + resIdxN];
      }}
    }}

    C += {BK};     // move BK columns to right
    // curOs += {BM} * {BK};
    __syncthreads();
  }}

  // // write out the results
  // for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
  //   for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
  //     C[(threadRow * {TM} + resIdxM) * K + threadCol * {TN} + resIdxN] =
  //         C[(threadRow * {TM} + resIdxM) * K + threadCol * {TN} + resIdxN] +
  //         threadResults[resIdxM * {TN} + resIdxN];
  //   }}
  // }}

  __syncthreads();

  }} // cCol

}}
'''


kernel_code_7 = '''
extern "C" __global__ void __launch_bounds__(({BM} * {BN}) / ({TM} * {TN}), 1)
    sgemm2DBlocktiling(int M, int N, int K,
                       const float *origA, const float *origB, float *origC, const float *origD) {{
  // const int cRow = blockIdx.y;
  // const int cCol = blockIdx.x;
  const int cRow = blockIdx.x;

  const int totalResultsBlocktile = {BM} * {BN};
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const int numThreadsBlocktile = totalResultsBlocktile / ({TM} * {TN});

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % ({BN} / {TN});
  const int threadRow = threadIdx.x / ({BN} / {TN});

  // allocate space for the current blocktile in smem
  __shared__ float As[{BM} * {BK}];
  __shared__ float Bs[{BK} * {BN}];
  // __shared__ float Cs[{BM} * {BN}];
  float *Cs = As;
  float *Ds = Bs;  // BN x BK
  // float *Os = As;  // BM x BK
  __shared__ float Os[{BM} * {BK} * 4];
  // __shared__ float Os[{BM} * {BK}];

  for (int i = threadIdx.x; i < {BM} * {BK} * 4; i += {BM} * {BK}) {{
    Os[i] = 0.0;
  }}

  const float *A;
  const float *B;
  float *C;
  const float *D;
  float *curOs;

  int blocksM = M / {BM};
  int blocksN = N / {BN};

  for (int cCol = 0; cCol < blocksN; ++cCol) {{
  
  A = origA;
  B = origB;
  C = origC;
  D = origD;

  // Move blocktile to beginning of A's row and B's column
  A += cRow * {BM} * K;
  B += cCol * {BN};
  C += cRow * {BM} * K;
  D += cCol * {BN} * K;  // Is this correct?

  // calculating the indices that this thread will load into SMEM
  const int innerRowA = threadIdx.x / {BK};
  const int innerColA = threadIdx.x % {BK};
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const int strideA = numThreadsBlocktile / {BK};
  const int innerRowB = threadIdx.x / {BN};
  const int innerColB = threadIdx.x % {BN};
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const int strideB = numThreadsBlocktile / {BN};
  const int innerRowD = threadIdx.x / {BK};
  const int innerColD = threadIdx.x % {BK};
  const int strideD = numThreadsBlocktile / {BK};

  // allocate thread-local cache for results in registerfile
  float threadResults[{TM} * {TN}] = {{0.0}};
  // register caches for As and Bs
  float regM[{TM}] = {{0.0}};
  float regN[{TN}] = {{0.0}};

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += {BK}) {{
    // populate the SMEM caches
    for (int loadOffset = 0; loadOffset < {BM}; loadOffset += strideA) {{
      As[(innerRowA + loadOffset) * {BK} + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }}
    for (int loadOffset = 0; loadOffset < {BK}; loadOffset += strideB) {{
      Bs[(innerRowB + loadOffset) * {BN} + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }}
    __syncthreads();

    // advance blocktile
    A += {BK};     // move BK columns to right
    B += {BK} * N; // move BK rows down
for (int loadOffset = 0; loadOffset < {BM}; loadOffset += strideD) {{
      curOs[(innerRowD + loadOffset) * {BK} + innerColD] =
          C[(innerRowD + loadOffset) * K + innerColD];
    }}
    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < {BK}; ++dotIdx) {{
      // block into registers
      for (int i = 0; i < {TM}; ++i) {{
        regM[i] = As[(threadRow * {TM} + i) * {BK} + dotIdx];
      }}
      for (int i = 0; i < {TN}; ++i) {{
        regN[i] = Bs[dotIdx * {BN} + threadCol * {TN} + i];
      }}
      for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
        for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
          threadResults[resIdxM * {TN} + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }}
      }}
    }}
    __syncthreads();
  }}

  // write out to SMEM
  for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
    for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
      Cs[(threadRow * {TM} + resIdxM) * {BN} + threadCol * {TN} + resIdxN] =
          threadResults[resIdxM * {TN} + resIdxN];
    }}
  }}

  // for (int i = 0; i < {TM} * {TN}; ++i) {{
  //   threadResults[i] = 0.0;
  // }}

  curOs = Os;

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += {BK}) {{
    // populate the SMEM caches
    for (int loadOffset = 0; loadOffset < {BK}; loadOffset += strideD) {{
      Ds[(innerRowD + loadOffset) * {BK} + innerColD] =
          D[(innerRowD + loadOffset) * K + innerColD];
    }}
    for (int loadOffset = 0; loadOffset < {BM}; loadOffset += strideD) {{
      curOs[(innerRowD + loadOffset) * {BK} + innerColD] =
          C[(innerRowD + loadOffset) * K + innerColD];
    }}
    __syncthreads();

    // advance blocktile
    D += {BK};     // move BK columns to right

    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < {BN}; ++dotIdx) {{
      // block into registers
      for (int i = 0; i < {TM}; ++i) {{
        regM[i] = Cs[(threadRow * {TM} + i) * {BN} + dotIdx];
      }}
      for (int i = 0; i < {TN}; ++i) {{
        regN[i] = Ds[dotIdx * {BK} + threadCol * {TN} + i];
      }}
      for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
        for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
          // threadResults[resIdxM * {TN} + resIdxN] = regM[resIdxM] * regN[resIdxN];
          curOs[(threadRow * {TM} + resIdxM) * {BK} + threadCol * {TN} + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }}
      }}
    }}

    for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
      for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
        // threadResults[resIdxM * {TN} + resIdxN] = regM[resIdxM] * regN[resIdxN];
        C[(threadRow * {TM} + resIdxM) * K + threadCol * {TN} + resIdxN] =
            curOs[(threadRow * {TM} + resIdxM) * {BK} + threadCol * {TN} + resIdxN];
      }}
    }}

    C += {BK};     // move BK columns to right
    curOs += {BM} * {BK};
    __syncthreads();
  }}

  // // write out the results
  // for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
  //   for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
  //     C[(threadRow * {TM} + resIdxM) * K + threadCol * {TN} + resIdxN] =
  //         C[(threadRow * {TM} + resIdxM) * K + threadCol * {TN} + resIdxN] +
  //         threadResults[resIdxM * {TN} + resIdxN];
  //   }}
  // }}

  __syncthreads();

  }} // cCol

}}
'''


kernel_code_8 = '''
extern "C" __global__ void __launch_bounds__(({BM} * {BN}) / ({TM} * {TN}), 1)
    sgemm2DBlocktiling(int M, int N, int K,
                       const float *origA, const float *origB, float *origC, const float *origD) {{
  // const int cRow = blockIdx.y;
  // const int cCol = blockIdx.x;
  const int cRow = blockIdx.x;

  const int totalResultsBlocktile = {BM} * {BN};
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const int numThreadsBlocktile = totalResultsBlocktile / ({TM} * {TN});

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % ({BN} / {TN});
  const int threadRow = threadIdx.x / ({BN} / {TN});

  // allocate space for the current blocktile in smem
  __shared__ float As[{BM} * {BK}];
  __shared__ float Bs[{BK} * {BN}];
  // __shared__ float Cs[{BM} * {BN}];
  float *Cs = As;
  float *Ds = Bs;  // BN x BK
  // float *Os = As;  // BM x BK
  // __shared__ float Os[{BM} * {BK} * 4];
  __shared__ float Os[{BM} * {BK}];

  // for (int i = threadIdx.x; i < {BM} * {BK} * 4; i += {BM} * {BK}) {{
  //   Os[i] = 0.0;
  // }}

  __shared__ float mi[{BM}];
  __shared__ float li[{BM}];

  const float *A;
  const float *B;
  float *C;
  const float *D;
  // float *curOs;

  int blocksM = M / {BM};
  int blocksN = N / {BN};

  for (int cCol = 0; cCol < blocksN; ++cCol) {{
  
  A = origA;
  B = origB;
  C = origC;
  D = origD;

  // Move blocktile to beginning of A's row and B's column
  A += cRow * {BM} * K;
  B += cCol * {BN};
  C += cRow * {BM} * K;
  D += cCol * {BN} * K;  // Is this correct?

  // calculating the indices that this thread will load into SMEM
  const int innerRowA = threadIdx.x / {BK};
  const int innerColA = threadIdx.x % {BK};
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const int strideA = numThreadsBlocktile / {BK};
  const int innerRowB = threadIdx.x / {BN};
  const int innerColB = threadIdx.x % {BN};
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const int strideB = numThreadsBlocktile / {BN};
  const int innerRowD = threadIdx.x / {BK};
  const int innerColD = threadIdx.x % {BK};
  const int strideD = numThreadsBlocktile / {BK};

  // allocate thread-local cache for results in registerfile
  float threadResults[{TM} * {TN}] = {{0.0}};
  // register caches for As and Bs
  float regM[{TM}] = {{0.0}};
  float regN[{TN}] = {{0.0}};

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += {BK}) {{
    // populate the SMEM caches
    for (int loadOffset = 0; loadOffset < {BM}; loadOffset += strideA) {{
      As[(innerRowA + loadOffset) * {BK} + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }}
    for (int loadOffset = 0; loadOffset < {BK}; loadOffset += strideB) {{
      Bs[(innerRowB + loadOffset) * {BN} + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }}
    __syncthreads();

    // advance blocktile
    A += {BK};     // move BK columns to right
    B += {BK} * N; // move BK rows down

    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < {BK}; ++dotIdx) {{
      // block into registers
      for (int i = 0; i < {TM}; ++i) {{
        regM[i] = As[(threadRow * {TM} + i) * {BK} + dotIdx];
      }}
      for (int i = 0; i < {TN}; ++i) {{
        regN[i] = Bs[dotIdx * {BN} + threadCol * {TN} + i];
      }}
      for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
        for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
          threadResults[resIdxM * {TN} + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }}
      }}
    }}
    __syncthreads();
  }}

  // write out to SMEM
  for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
    for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
      Cs[(threadRow * {TM} + resIdxM) * {BN} + threadCol * {TN} + resIdxN] =
          threadResults[resIdxM * {TN} + resIdxN];
    }}
  }}

  // for (int i = 0; i < {TM} * {TN}; ++i) {{
  //   threadResults[i] = 0.0;
  // }}

  // curOs = Os;

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += {BK}) {{
    // populate the SMEM caches
    for (int loadOffset = 0; loadOffset < {BK}; loadOffset += strideD) {{
      Ds[(innerRowD + loadOffset) * {BK} + innerColD] =
          D[(innerRowD + loadOffset) * K + innerColD];
    }}
    for (int loadOffset = 0; loadOffset < {BM}; loadOffset += strideD) {{
      Os[(innerRowD + loadOffset) * {BK} + innerColD] =
          C[(innerRowD + loadOffset) * K + innerColD];
    }}
    __syncthreads();

    // advance blocktile
    D += {BK};     // move BK columns to right

    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < {BN}; ++dotIdx) {{
      // block into registers
      for (int i = 0; i < {TM}; ++i) {{
        regM[i] = Cs[(threadRow * {TM} + i) * {BN} + dotIdx];
      }}
      for (int i = 0; i < {TN}; ++i) {{
        regN[i] = Ds[dotIdx * {BK} + threadCol * {TN} + i];
      }}
      for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
        for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
          // threadResults[resIdxM * {TN} + resIdxN] = regM[resIdxM] * regN[resIdxN];
          Os[(threadRow * {TM} + resIdxM) * {BK} + threadCol * {TN} + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }}
      }}
    }}

    for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
      for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
        // threadResults[resIdxM * {TN} + resIdxN] = regM[resIdxM] * regN[resIdxN];
        C[(threadRow * {TM} + resIdxM) * K + threadCol * {TN} + resIdxN] =
            Os[(threadRow * {TM} + resIdxM) * {BK} + threadCol * {TN} + resIdxN];
      }}
    }}

    C += {BK};     // move BK columns to right
    // curOs += {BM} * {BK};
    __syncthreads();
  }}

  // // write out the results
  // for (int resIdxM = 0; resIdxM < {TM}; ++resIdxM) {{
  //   for (int resIdxN = 0; resIdxN < {TN}; ++resIdxN) {{
  //     C[(threadRow * {TM} + resIdxM) * K + threadCol * {TN} + resIdxN] =
  //         C[(threadRow * {TM} + resIdxM) * K + threadCol * {TN} + resIdxN] +
  //         threadResults[resIdxM * {TN} + resIdxN];
  //   }}
  // }}

  __syncthreads();

  }} // cCol

}}
'''


def rel_error(val, ref):
    return cp.linalg.norm(val - ref) / cp.linalg.norm(ref)


if __name__ == "__main__":

    rng = np.random.default_rng(0)

    batch = 10

    # M, N, K = 1024, 1024, 1024
    # M, N, K = 2048, 2048, 2048
    M, N, K = 8192, 8192, 128
    # M, N, K, = 16384, 16384, 128
    A_host = rng.random((M, K)).astype(np.float32)
    B_host = rng.random((K, N)).astype(np.float32)
    C_host = rng.random((M, N)).astype(np.float32)
    D_host = rng.random((N, K)).astype(np.float32)

    alpha = np.float32(1.0)
    beta = np.float32(0.0)

    A_dev = cp.asarray(A_host)
    B_dev = cp.asarray(B_host)
    C_dev = cp.asarray(C_host)
    D_dev = cp.asarray(D_host)

    ref = A_dev @ B_dev

    BM, BN, BK = 64, 64, 64
    TM, TN = 4, 4
    code_4 = kernel_code_5.format(BM=BM, BN=BN, BK=BK, TM=TM, TN=TN)
    kernel_4 = cp.RawKernel(code_4, "sgemm2DBlocktiling")

    gridDim = (math.ceil(M / BM),)
    blockDim = ((BM * BN) // (TM * TN), )
    print(gridDim, blockDim)
    val = cp.zeros_like(C_dev)
    kernel_4(gridDim, blockDim, (M, N, K, A_dev, B_dev, val))
    cp.cuda.Stream.null.synchronize()

    print(rel_error(val, ref))
    print(cp.allclose(val, ref))

    def _func():
        kernel_4(gridDim, blockDim, (M, N, K, A_dev, B_dev, val))
        cp.cuda.Stream.null.synchronize()
      
    runtimes = []
    for _ in range(50):
        start = time.time()
        _func()
        runtimes.append(time.time() - start)
    flops = 2 * M * N * K
    gflops_s = flops / (1e9 * np.median(runtimes))
    runtime = np.median(runtimes) * 1000
    print(f"Runtime: {runtime:.2f} ms, Gflops/s: {gflops_s:.2f}")

    ref = A_dev @ B_dev @ D_dev

    BM, BN, BK = 64, 64, 64
    TM, TN = 4, 4
    code_4 = kernel_code_6.format(BM=BM, BN=BN, BK=BK, TM=TM, TN=TN)
    kernel_4 = cp.RawKernel(code_4, "sgemm2DBlocktiling")

    gridDim = (math.ceil(M / BM), )
    blockDim = ((BM * BN) // (TM * TN), )
    print(gridDim, blockDim)
    val = cp.zeros_like(ref)
    kernel_4(gridDim, blockDim, (M, N, K, A_dev, B_dev, val, D_dev))
    cp.cuda.Stream.null.synchronize()

    print(rel_error(val, ref))
    print(cp.allclose(val, ref))

    def _func():
        kernel_4(gridDim, blockDim, (M, N, K, A_dev, B_dev, val, D_dev))
        cp.cuda.Stream.null.synchronize()
      
    runtimes = []
    for _ in range(50):
        start = time.time()
        _func()
        runtimes.append(time.time() - start)
    flops = 4 * M * N * K
    gflops_s = flops / (1e9 * np.median(runtimes))
    runtime = np.median(runtimes) * 1000
    print(f"Runtime: {runtime:.2f} ms, Gflops/s: {gflops_s:.2f}")

    BM, BN, BK = 32, 32, 32
    TM, TN = 2, 2
    code_4 = kernel_code_7.format(BM=BM, BN=BN, BK=BK, TM=TM, TN=TN)
    kernel_4 = cp.RawKernel(code_4, "sgemm2DBlocktiling")

    gridDim = (math.ceil(M / BM), )
    blockDim = ((BM * BN) // (TM * TN), )
    print(gridDim, blockDim)
    val = cp.zeros_like(ref)
    kernel_4(gridDim, blockDim, (M, N, K, A_dev, B_dev, val, D_dev))
    cp.cuda.Stream.null.synchronize()

    print(rel_error(val, ref))
    print(cp.allclose(val, ref))

    def _func():
        kernel_4(gridDim, blockDim, (M, N, K, A_dev, B_dev, val, D_dev))
        cp.cuda.Stream.null.synchronize()
      
    runtimes = []
    for _ in range(50):
        start = time.time()
        _func()
        runtimes.append(time.time() - start)
    flops = 4 * M * N * K
    gflops_s = flops / (1e9 * np.median(runtimes))
    runtime = np.median(runtimes) * 1000
    print(f"Runtime: {runtime:.2f} ms, Gflops/s: {gflops_s:.2f}")

    exit(0)

    ref = A_dev @ B_dev @ D_dev
    val = cp.zeros_like(ref)

    BM, BN = 64, 64
    NBM, NBN = M // BM, N // BN

    for i in range(NBM):
        for j in range(NBN):
            Cij = A_dev[i * BM:(i + 1) * BM] @ B_dev[:, j * BN:(j + 1) * BN]
            val[i * BM:(i + 1) * BM, :] += Cij @ D_dev[j * BN:(j + 1) * BN, :]
    
    print(rel_error(val, ref))
    print(cp.allclose(val, ref))

    exit(0)

    ref = alpha * A_dev @ B_dev + beta * C_dev
    cp.cuda.Stream.null.synchronize()

    BM, BN, BK = 64, 64, 32
    TM, TN = 4, 4

    code = kernel_code.format(BM=BM, BN=BN, BK=BK, TM=TM, TN=TN)
    # kernel = cp.RawKernel(code, "sgemm2DBlocktiling", backend="nvcc", options=("--std=c++14",))
    kernel = cp.RawKernel(code, "sgemm2DBlocktiling")

    gridDim = (math.ceil(N / BN), math.ceil(M / BM))
    blockDim = ((BM * BN) // (TM * TN), )
    print(gridDim, blockDim)
    val = cp.copy(C_dev)
    kernel(gridDim, blockDim, (M, N, K, alpha, A_dev, B_dev, beta, val))
    cp.cuda.Stream.null.synchronize()

    print(rel_error(val, ref))    
    print(cp.allclose(val, ref))

    def _func():
        kernel(gridDim, blockDim, (M, N, K, alpha, A_dev, B_dev, beta, val))
        cp.cuda.Stream.null.synchronize()

    runtimes = []
    for _ in range(50):
        start = time.time()
        _func()
        runtimes.append(time.time() - start)
    flops = 2 * M * N * K
    gflops_s = flops / (1e9 * np.median(runtimes))
    runtime = np.median(runtimes) * 1000
    print(f"Runtime: {runtime:.2f} ms, Gflops/s: {gflops_s:.2f}")

    code_2 = kernel_code_2.format(BM=BM, BN=BN, BK=BK, TM=TM, TN=TN)
    kernel_2 = cp.RawKernel(code_2, "sgemm2DBlocktiling")

    gridDim = (1, 1)
    blockDim = ((BM * BN) // (TM * TN), )
    print(gridDim, blockDim)
    val_2 = cp.copy(C_dev)
    kernel_2(gridDim, blockDim, (M, N, K, alpha, A_dev, B_dev, beta, val_2))
    cp.cuda.Stream.null.synchronize()

    print(rel_error(val_2, ref))
    print(cp.allclose(val_2, ref))

    def _func():
        kernel_2(gridDim, blockDim, (M, N, K, alpha, A_dev, B_dev, beta, val_2))
        cp.cuda.Stream.null.synchronize()

    runtimes = []
    for _ in range(50):
        start = time.time()
        _func()
        runtimes.append(time.time() - start)
    flops = 2 * M * N * K
    gflops_s = flops / (1e9 * np.median(runtimes))
    runtime = np.median(runtimes) * 1000
    print(f"Runtime: {runtime:.2f} ms, Gflops/s: {gflops_s:.2f}")


    BM, BN, BK = 64, 64, K
    TM, TN = 4, 4
    code_3 = kernel_code_3.format(BM=BM, BN=BN, BK=BK, BL=BK, TM=TM, TN=TN)
    kernel_3 = cp.RawKernel(code_3, "sgemm2DBlocktiling")

    gridDim = (1, 1)
    blockDim = ((BM * BN) // (TM * TN), )
    print(gridDim, blockDim)
    val_3 = cp.copy(C_dev)
    kernel_3(gridDim, blockDim, (M, N, K, alpha, A_dev, B_dev, beta, val_3))
    cp.cuda.Stream.null.synchronize()

    print(rel_error(val_3, ref))
    print(cp.allclose(val_3, ref))

    def _func():
        kernel_3(gridDim, blockDim, (M, N, K, alpha, A_dev, B_dev, beta, val_3))
        cp.cuda.Stream.null.synchronize()

    runtimes = []
    for _ in range(50):
        start = time.time()
        _func()
        runtimes.append(time.time() - start)
    flops = 2 * M * N * K
    gflops_s = flops / (1e9 * np.median(runtimes))
    runtime = np.median(runtimes) * 1000
    print(f"Runtime: {runtime:.2f} ms, Gflops/s: {gflops_s:.2f}")

