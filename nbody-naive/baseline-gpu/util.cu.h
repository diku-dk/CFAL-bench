#ifndef SCAN_KERS
#define SCAN_KERS

#include <cuda_runtime.h>

#define WARP    32
#define lgWARP  5

/**
 * Generic Add operator that can be instantiated over
 *  numeric-basic types, such as int32_t, int64_t,
 *  float, double, etc.
 */
template<class T>
class Add {
  public:
    typedef T InpElTp;
    typedef T RedElTp;
    static const bool commutative = true;
    static __device__ __host__ inline T identInp()                    { return (T)0;    }
    static __device__ __host__ inline T mapFun(const T& el)           { return el;      }
    static __device__ __host__ inline T identity()                    { return (T)0;    }
    static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }

    static __device__ __host__ inline bool equals(const T t1, const T t2) { return (t1 == t2); }
    static __device__ __host__ inline T remVolatile(volatile T& t)    { T res = t; return res; }
};

#if 0
class AddD3 {
  public:
    typedef double3 InpElTp;
    typedef double3 RedElTp;
    typedef double3 T;
    static const bool commutative = true;
    static __device__ __host__ inline T identInp()                    { return make_double3(0.0, 0.0, 0.0); }
    static __device__ __host__ inline T mapFun(const T& el)           { return el;      }
    static __device__ __host__ inline T identity()                    { return make_double3(0.0, 0.0, 0.0); }
    static __device__ __host__ inline T apply(const T t1, const T t2) {
      double3 res;
      res.x = t1.x + t2.x;
      res.y = t1.y + t2.y;
      res.z = t1.z + t2.z; 
      return res;
    }
    static __device__ __host__ inline bool equals(const T t1, const T t2) {
      return ( (t1.x == t2.x) && (t1.y == t2.y) && (t1.z == t2.z) ); 
    }
    static __device__ __host__ inline T remVolatile(volatile T& t) { 
      double3 res;
      res.x = t.x; 
      res.y = t.y;
      res.z = t.z;
      return res; 
    }
};
#endif

class D3 {
  public:
    double x; double y; double z;
    __device__ __host__ inline D3() { x = 0; y = 0; z = 0; } 
    __device__ __host__ inline D3(const double& a, const double& b, const double& c) { x = a; y = b; z = c; }
    __device__ __host__ inline D3(const D3& d3) { x = d3.x; y = d3.y; z = d3.z; }
    __device__ __host__ inline void operator=(const D3& d3) volatile { x = d3.x; y = d3.y; z = d3.z; }
};

/**
 * Representation of the pairwise integer plus operator
 */
class AddD3 {
  public:
    typedef D3 RedElTp;
    static __device__ __host__ inline D3 identity() { return D3(0,0,0); }
    static __device__ __host__ inline D3 apply(volatile D3& t1, volatile D3& t2) {
        return D3(t1.x + t2.x, t1.y + t2.y, t1.z + t2.z);
    }
    static __device__ __host__ inline D3 remVolatile(volatile D3& t) {
        D3 res; res.x = t.x; res.y = t.y; res.z = t.z; return res;
    }
    static __device__ __host__ inline bool equals(D3& t1, D3& t2) {
        return (t1.x == t2.x && t1.y == t2.y && t1.z == t2.z);
    }
};


/***************************************/
/*** Scan Inclusive Helpers & Kernel ***/
/***************************************/

/**
 * A warp of threads cooperatively scan with generic-binop `OP` a 
 *   number of warp elements stored in shared memory (`ptr`).
 * No synchronization is needed because the thread in a warp execute
 *   in lockstep.
 * `idx` is the local thread index within a cuda block (threadIdx.x)
 * Each thread returns the corresponding scanned element of type
 *   `typename OP::RedElTp`
 */ 
template<class OP>
__device__ inline typename OP::RedElTp
scanIncWarp( volatile typename OP::RedElTp* ptr, const unsigned int idx ) {
    const unsigned int lane = idx & (WARP-1);
    #pragma unroll
    for(uint32_t i=0; i<lgWARP; i++) {
        const uint32_t p = (1<<i);
        if( lane >= p ) ptr[idx] = OP::apply(ptr[idx-p], ptr[idx]);
        __syncwarp();
    }
    return OP::remVolatile(ptr[idx]);
}

/**
 * A CUDA-block of threads cooperatively scan with generic-binop `OP`
 *   a CUDA-block number of elements stored in shared memory (`ptr`).
 * `idx` is the local thread index within a cuda block (threadIdx.x)
 * Each thread returns the corresponding scanned element of type
 *   `typename OP::RedElTp`. Note that this is NOT published to shared memory!
 */ 
template<class OP>
__device__ inline typename OP::RedElTp
scanIncBlock(volatile typename OP::RedElTp* ptr, const unsigned int idx) {
    const unsigned int lane   = idx & (WARP-1);
    const unsigned int warpid = idx >> lgWARP;

    // 1. perform scan at warp level
    typename OP::RedElTp res = scanIncWarp<OP>(ptr,idx);
    __syncthreads();

    // 2. place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and 
    //   max block size = 32^2 = 1024
    if (lane == (WARP-1)) { ptr[warpid] = res; } 
    __syncthreads();

    // 3. scan again the first warp
    if (warpid == 0) scanIncWarp<OP>(ptr, idx);
    __syncthreads();

    // 4. accumulate results from previous step;
    if (warpid > 0) {
        res = OP::apply(ptr[warpid-1], res);
    }

    return res;
}

#endif //SCAN_KERS

