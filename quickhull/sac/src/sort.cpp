#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>

extern "C" {
    /* All these arrays live on the device */
    void MySort(int n, void *array, int size, double *keys, void *array_out)
    {
        int *indices = (int *)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
    
        /* This gives the index permutation */
        thrust::sort_by_key(thrust::host, keys, keys + n, indices);
    
        /* Permute the array */
        /* Actually this is array_out[i] -> array[indices[iv]] */
        for (int i = 0; i < n; i++) {
            memcpy((void *)((uintptr_t)array_out + i * size), 
                   (void *)((uintptr_t)array + indices[i] * size),
                   size);
        }

        free(indices);
    }

    void MySortDouble(double *array, double *keys, int elem_per_slice, 
                      int n)
    {
        double *array_out = (double *)malloc(n * elem_per_slice * 
                                             sizeof(double));
        MySort(n, array, elem_per_slice * sizeof(double), keys, array_out);
        memcpy((void *)array, (void *)array_out, 
               elem_per_slice * n * sizeof(double));
        free(array_out);
    }
}
