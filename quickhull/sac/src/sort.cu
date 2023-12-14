#include <thrust/sort.h>

#define THREADS 1024

__global__ void InitInd(size_t n, size_t *indices)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        indices[tid] = n;
    }
}

__global__ void Permute(size_t n, void *array, size_t size, size_t *indices, 
                        void *array_out)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        for (int i = 0; i < size; i++) {
            ((char *)array_out)[tid * size + i] = 
                    ((char *)array)[indices[tid] * size + i];
        }
    }
}

extern "C" {
    /* All these arrays live on the device */
    void MySort(size_t n, void *array, size_t size, double *keys, void *array_out)
    {
        size_t *indices;
        cudaMalloc(&indices, n * sizeof(size_t));
        int blocks = (n + THREADS - 1) / THREADS;
        InitInd<<<blocks, THREADS>>>(n, indices);
    
        /* This gives the index permutation */
        thrust::sort_by_key(thrust::device, keys, keys + n, indices);
    
        /* Permute the array */
        Permute<<<blocks, THREADS>>>(n, array, size, indices, array_out);

        cudaFree(indices);
    }

    void MySortDouble(double *array, double *keys, int elem_per_slice, 
                      int n)
    {
        double *array_out;
        cudaMalloc(&array_out, n * elem_per_slice * sizeof(double));
        MySort(n, array, elem_per_slice * sizeof(double), keys, array_out);
        cudaMemcpy(array, array_out, n * elem_per_slice * sizeof(double),
                   cudaMemcpyDeviceToDevice);
    }
}
