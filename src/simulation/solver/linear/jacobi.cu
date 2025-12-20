#include <linear/jacobi.h>
#include <linear/cuUtils.cuh>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>

template<typename T>
inline JacobiSolver<T>::JacobiSolver(int N, int maxIter) : maxIter(maxIter)
{
    cudaMalloc((void**)&d_rowPtrA, sizeof(int) * (N + 1));
    CHECK_CUDA(cudaMalloc((void**)&x_prime, N * sizeof(T)));
    CHECK_CUSPARSE(cusparseCreate(&cusHandle));
    this->capacity = 0;
}

template<typename T>
JacobiSolver<T>::~JacobiSolver()
{
    cudaFree(x_prime);
    CHECK_CUSPARSE(cusparseDestroy(cusHandle));
}

template<typename T>
__global__ void JacobiCSRKernel(int N, const T* __restrict__ d_b, T* __restrict__ d_x_new, 
                                const T* __restrict__ d_A, int nz, const int* __restrict__ d_rowPtr, 
                                const int* __restrict__ d_colIdx, const T* __restrict__ d_x_old) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        T sum = 0.0f;
        T diag = 1.0f;
        int row_start = d_rowPtr[idx];
        int row_end = d_rowPtr[idx + 1];

        for (int i = row_start; i < row_end; ++i) {
            int col = d_colIdx[i];
            T val = d_A[i];            
            if (col == idx)
                diag = val;
            else 
                sum -= val * d_x_old[col];
        }
        
    
        if (abs(diag) > 1e-15) {
            d_x_new[idx] = (d_b[idx] + sum) / diag;
        } else {
            d_x_new[idx] = d_x_old[idx];
        }
    }
}

template<typename T>
void JacobiSolver<T>::Solve(int N, T* d_b, T* d_x, T* A, int nz, int* rowIdx, int* colIdx, T* d_guess)
{
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    sort_coo(N, nz, A, rowIdx, colIdx, this->d_A, this->d_rowIdx, this->d_colIdx, this->capacity);
    CHECK_CUSPARSE(cusparseXcoo2csr(cusHandle, this->d_rowIdx, nz, N, d_rowPtrA, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUDA(cudaMemset(d_x, 0, N * sizeof(T)));
    
    T* current_x = d_x;
    T* next_x = x_prime;

    T err{ 100.0 };

    for (size_t i = 0; i < maxIter && err > 1e-3; i++)
    {    
        JacobiCSRKernel<<<numBlocks, blockSize>>>(N, d_b, next_x, this->d_A, nz, d_rowPtrA, this->d_colIdx, current_x);    
        thrust::device_ptr<T> ptr_next(next_x);
        thrust::device_ptr<T> ptr_curr(current_x);
        
        err = thrust::transform_reduce(
            thrust::counting_iterator<indexType>(0),
            thrust::counting_iterator<indexType>(N),
            [=]__host__ __device__(indexType vertIdx) {
                T diff = ptr_next[vertIdx] - ptr_curr[vertIdx];
                return diff * diff;
            },
            0.0,
            thrust::plus<T>()) / N;
    
        std::swap(current_x, next_x);
    }


    if (current_x != d_x) {
        CHECK_CUDA(cudaMemcpy(d_x, current_x, N * sizeof(T), cudaMemcpyDeviceToDevice));
    }
}

template class JacobiSolver<float>;
template class JacobiSolver<double>;