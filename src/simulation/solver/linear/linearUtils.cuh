#pragma once
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}
#define CHECK_CUSOLVER(func)                                                   \
{                                                                              \
    cusolverStatus_t status = (func);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        printf("cuSOLVER API failed at line %d with error: %s (%d)\n",         \
               __LINE__, "error", status);              \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
    }                                                                          \
}

template<typename T>
void sort_coo(int N, int &nz, T* d_A, int* d_rowIdx, int* d_colIdx, T*& new_A, int*& new_rowIdx, int*& new_colIdx) {
    cudaMalloc(&new_rowIdx, nz * sizeof(int));
    cudaMalloc(&new_colIdx, nz * sizeof(int));
    cudaMalloc(&new_A, nz * sizeof(T));
    
    thrust::device_ptr<int> d_row(d_rowIdx);
    thrust::device_ptr<int> d_col(d_colIdx);
    thrust::device_ptr<T> d_val(d_A);

    thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(d_row, d_col, d_val)),
                 thrust::make_zip_iterator(thrust::make_tuple(d_row + nz, d_col + nz, d_val + nz)));
    thrust::device_ptr<int> d_new_row(new_rowIdx);
    thrust::device_ptr<int> d_new_col(new_colIdx);
    thrust::device_ptr<T> d_new_val(new_A);

    int new_nz = thrust::reduce_by_key(
        thrust::make_zip_iterator(thrust::make_tuple(d_row, d_col)),
        thrust::make_zip_iterator(thrust::make_tuple(d_row, d_col)) + nz,
        d_val,
        thrust::make_zip_iterator(thrust::make_tuple(d_new_row, d_new_col)),
        d_new_val,
        thrust::equal_to< thrust::tuple<int, int> >(),
        thrust::plus<T>()
    ).first - thrust::make_zip_iterator(thrust::make_tuple(d_new_row, d_new_col));
    nz = new_nz;
}
