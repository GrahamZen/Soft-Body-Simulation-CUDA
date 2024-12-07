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
}

template<typename T>
JacobiSolver<T>::~JacobiSolver()
{
    cudaFree(d_rowPtrA);
    cudaFree(x_prime);
    CHECK_CUSPARSE(cusparseDestroy(cusHandle));
}

template<typename T>
__global__ void JacobiCOOKernel(int N, T* d_b, T* d_x, T* A, int nz, int* rowIdx, int* colIdx, T* d_guess) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        T sum = 0.0f;
        T diag = 0.0f;
        for (int i = 0; i < nz; ++i) {
            if (rowIdx[i] == idx) {
                if (colIdx[i] == idx) {
                    diag = A[i];
                }
                else {
                    sum -= A[i] * d_x[colIdx[i]];
                }
            }
        }
        if (diag != 0.0f) {
            d_x[idx] = (d_b[idx] + sum) / diag;
        }
    }
}

template<typename T>
__global__ void JacobiCSRKernel(int N, T* d_b, T* d_x, T* d_A, int nz, int* d_rowPtr, int* d_colIdx, T* d_guess) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        T sum = 0.0f;
        T diag = 0.0f;
        int row_start = d_rowPtr[idx];
        int row_end = d_rowPtr[idx + 1];

        for (int i = row_start; i < row_end; ++i) {
            int col = d_colIdx[i];
            if (col == idx) {
                diag = d_A[i];
            }
            else {
                sum -= d_A[i] * d_guess[col];
            }
        }
        if (diag != 0.0f) {
            d_x[idx] = (d_b[idx] + sum) / diag;
        }
    }
}

template<typename T>
void JacobiSolver<T>::Solve(int N, T* d_b, T* d_x, T* A, int nz, int* rowIdx, int* colIdx, T* d_guess)
{
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    sort_coo(N, nz, A, rowIdx, colIdx, d_A, d_rowIdx, d_colIdx);
    CHECK_CUSPARSE(cusparseXcoo2csr(cusHandle, d_rowIdx, nz, N, d_rowPtrA, CUSPARSE_INDEX_BASE_ZERO));

    CHECK_CUDA(cudaMemset(d_x, 0, N * sizeof(T)));
    thrust::device_ptr<T> x_prime_ptr(x_prime);
    thrust::device_ptr<T> x_ptr(d_x);
    T err{ 1 };
    while (abs(err) > 1e-3) {
        JacobiCSRKernel << <numBlocks, blockSize >> > (N, d_b, x_prime, A, nz, d_rowPtrA, d_colIdx, d_x);
        err = thrust::transform_reduce(
            thrust::counting_iterator<indexType>(0),
            thrust::counting_iterator<indexType>(N),
            [=]__host__ __device__(indexType vertIdx) {
            return (x_prime_ptr[vertIdx] - x_ptr[vertIdx]) * (x_prime_ptr[vertIdx] - x_ptr[vertIdx]);
        },
            0.0,
            thrust::plus<T>()) / N;
        cudaMemcpy(d_x, x_prime, N * sizeof(T), cudaMemcpyDeviceToDevice);
    }
}

template class JacobiSolver<float>;
template class JacobiSolver<double>;