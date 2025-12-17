#pragma once
#include <def.h>
#include <library_types.h>

template <typename T>
struct CUDAType {
    static constexpr cudaDataType value = CUDA_R_32F;  //
};

template <>
struct CUDAType<double> {
    static constexpr cudaDataType value = CUDA_R_64F;
};

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
class LinearSolver {
public:
    LinearSolver() = default;
    virtual ~LinearSolver() {
        if (d_A) cudaFree(d_A);
        if (d_rowIdx) cudaFree(d_rowIdx);
        if (d_colIdx) cudaFree(d_colIdx);
        if (d_rowPtrA) cudaFree(d_rowPtrA);
    }
    virtual void Solve(int N, T* d_b, T* d_x, T* A = nullptr, int nz = 0, int* rowIdx = nullptr, int* colIdx = nullptr, T* d_guess = nullptr) = 0;
protected:
    cudaDataType dType = CUDAType<T>::value;
    T* d_A = nullptr;
    int* d_rowIdx = nullptr;
    int* d_colIdx = nullptr;
    int* d_rowPtrA = nullptr; // CSR 
    int capacity = 0;
};
template<typename T>
void sort_coo(int N, int& nz, T* d_A, int* d_rowIdx, int* d_colIdx, T*& new_A, int*& new_rowIdx, int*& new_colIdx, int& capacity);