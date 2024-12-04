#include <linear/jacobi.h>
#include <linear/cuUtils.cuh>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>

template<typename T>
inline JacobiSolver<T>::JacobiSolver(int N, int maxIter) : maxIter(maxIter)
{
    cudaMalloc((void**)&d_rowPtrA, sizeof(int) * (N + 1));
    CHECK_CUDA(cudaMalloc((void**)&d_r, N * sizeof(T)));
    CHECK_CUDA(cudaMemset(d_r, 0, N * sizeof(T)));
    CHECK_CUBLAS(cublasCreate(&cubHandle));
    CHECK_CUSPARSE(cusparseCreate(&cusHandle));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_r, N, d_r, CUDA_R_64F));
}

template<typename T>
JacobiSolver<T>::~JacobiSolver()
{
    cudaFree(d_rowPtrA);
    CHECK_CUBLAS(cublasDestroy(cubHandle));
    CHECK_CUSPARSE(cusparseDestroy(cusHandle));
    CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_r));
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
                sum -= d_A[i] * d_x[col];
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
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_x, N, d_x, dType));
    CHECK_CUSPARSE(cusparseCreateCsr(&d_matA, N, N, nz, d_rowPtrA, d_colIdx, d_A,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dType));
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d_matA,
        dvec_x, &negone, dvec_r, dType, CUSPARSE_SPMV_CSR_ALG1, &bufferSize));
    CHECK_CUDA(cudaMalloc((void**)&d_buf, bufferSize));
    T max_residual{ 1 };
    while (max_residual > 1e-3) {
        JacobiCSRKernel << <numBlocks, blockSize >> > (N, d_b, d_x, A, nz, d_rowPtrA, d_colIdx, d_guess);
        CHECK_CUDA(cudaMemcpy(d_r, d_b, N * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUSPARSE(cusparseSpMV(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d_matA,
            dvec_x, &negone, dvec_r, dType, CUSPARSE_SPMV_CSR_ALG1, d_buf));
        max_residual = thrust::transform_reduce(
            thrust::device_pointer_cast(d_r),
            thrust::device_pointer_cast(d_r) + N,
            [=]__host__ __device__(T i) {
            return abs(i);
        },
            0.0,
            thrust::maximum<T>());
    }
    CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_x));
    CHECK_CUSPARSE(cusparseDestroySpMat(d_matA));
    CHECK_CUDA(cudaFree(d_buf));
}

template class JacobiSolver<float>;
template class JacobiSolver<double>;