#include <linear/pcgJacobi.h>
#include <linear/cuUtils.cuh>
#include <iostream>
#include <utilities.cuh>

template<typename T>
__global__ void ExtractInverseDiagonalKernel(int N, const T* A, const int* rowPtr, const int* colIdx, T* invDiag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        T diagVal = 1.0;
        int start = rowPtr[idx];
        int end = rowPtr[idx + 1];
        for (int i = start; i < end; ++i) {
            if (colIdx[i] == idx) {
                diagVal = A[i];
                break;
            }
        }

        // Avoid division by zero
        if (abs(diagVal) < 1e-12) diagVal = 1.0;
        invDiag[idx] = 1.0 / diagVal;
    }
}

template<typename T>
__global__ void ApplyJacobiPreconditionerKernel(int N, const T* r, const T* invDiag, T* z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        z[idx] = r[idx] * invDiag[idx];
    }
}

template<typename T>
PCGJacobiSolver<T>::PCGJacobiSolver(int N, int max_iter, T tolerance) : N(N), max_iter(max_iter), tolerance(tolerance)
{
    CHECK_CUBLAS(cublasCreate(&cubHandle));
    CHECK_CUSPARSE(cusparseCreate(&cusHandle));

    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    CHECK_CUDA(cudaMalloc((void**)&d_z, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&d_r, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&d_q, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&d_p, N * sizeof(T)));

    CHECK_CUDA(cudaMalloc((void**)&d_invDiag, N * sizeof(T)));

    CHECK_CUDA(cudaMalloc((void**)&d_rowPtrA, (N + 1) * sizeof(int)));

    CHECK_CUDA(cudaMemset(d_z, 0, N * sizeof(T)));
    CHECK_CUDA(cudaMemset(d_r, 0, N * sizeof(T)));
    CHECK_CUDA(cudaMemset(d_q, 0, N * sizeof(T)));
    CHECK_CUDA(cudaMemset(d_p, 0, N * sizeof(T)));

    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_p, N, d_p, dType));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_q, N, d_q, dType));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_x, N, d_x, dType));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_b, N, d_b, dType));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_r, N, d_r, dType));
}

template<typename T>
PCGJacobiSolver<T>::~PCGJacobiSolver()
{
    if (d_z) CHECK_CUDA(cudaFree(d_z));
    if (d_r) CHECK_CUDA(cudaFree(d_r));
    if (d_q) CHECK_CUDA(cudaFree(d_q));
    if (d_p) CHECK_CUDA(cudaFree(d_p));
    if (d_invDiag) CHECK_CUDA(cudaFree(d_invDiag));
    if (d_bufMV) CHECK_CUDA(cudaFree(d_bufMV));

    CHECK_CUBLAS(cublasDestroy(cubHandle));
    CHECK_CUSPARSE(cusparseDestroy(cusHandle));

    CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_p));
    CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_q));

    if (dvec_x) CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_x));
    if (dvec_b) CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_b));
    if (dvec_r) CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_r));

    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
}

template<typename T>
void PCGJacobiSolver<T>::Solve(int N, T* d_b, T* d_x, T* A, int nz, int* rowIdx, int* colIdx, T* d_guess)
{
    sort_coo(N, nz, A, rowIdx, colIdx, this->d_A, this->d_rowIdx, this->d_colIdx, this->capacity);
    CHECK_CUSPARSE(cusparseXcoo2csr(cusHandle, this->d_rowIdx, nz, N, this->d_rowPtrA, CUSPARSE_INDEX_BASE_ZERO));

    CHECK_CUSPARSE(cusparseCreateCsr(&d_matA, N, N, nz, this->d_rowPtrA, this->d_colIdx, this->d_A,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dType));

    if (dvec_x) CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_x));
    if (dvec_b) CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_b));
    if (dvec_r) CHECK_CUSPARSE(cusparseDestroyDnVec(dvec_r));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_x, N, d_x, dType));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_b, N, d_b, dType));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dvec_r, N, d_r, dType));

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d_matA,
        dvec_p, &zero, dvec_q, dType, CUSPARSE_SPMV_CSR_ALG1, &bufferSize));

    if (bufferSize > old_bufferSizeMV) {
        if (d_bufMV) CHECK_CUDA(cudaFree(d_bufMV));
        CHECK_CUDA(cudaMalloc(&d_bufMV, bufferSize));
        old_bufferSizeMV = bufferSize;
    }
    if (d_guess != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(d_x, d_guess, N * sizeof(T), cudaMemcpyDeviceToDevice));
        // r = b - A*x
        // q = A*x
        CHECK_CUSPARSE(cusparseSpMV(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d_matA,
            dvec_x, &zero, dvec_q, dType, CUSPARSE_SPMV_CSR_ALG1, d_bufMV));
        CHECK_CUDA(cudaMemcpy(d_r, d_b, N * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasAxpy(cubHandle, N, (T)-1, d_q, 1, d_r, 1));
    }
    else
    {
        CHECK_CUDA(cudaMemset(d_x, 0, N * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(d_r, d_b, N * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    ExtractInverseDiagonalKernel << <blocks, threads >> > (N, this->d_A, this->d_rowPtrA, this->d_colIdx, d_invDiag);

    for (k = 0; k < max_iter; ++k)
    {
        // Check residual
        CHECK_CUBLAS(cublasnrm2(cubHandle, N, d_r, 1, &rTr));
        if (rTr < tolerance) break;
        ApplyJacobiPreconditionerKernel << <blocks, threads >> > (N, d_r, d_invDiag, d_z);
        rho_t = rho;
        // rho = rk * zk
        CHECK_CUBLAS(cublasdot(cubHandle, N, d_r, 1, d_z, 1, &rho));
        if (abs(rho) < 1e-15) break;

        if (k == 0)
        {
            // pk = zk
            CHECK_CUBLAS(cublascopy(cubHandle, N, d_z, 1, d_p, 1));
        }
        else
        {
            // beta = (rk*zk) / (r{k-1}*z{k-1})
            beta = rho / rho_t;
            // pk = zk + beta*p{k-1}
            CHECK_CUBLAS(cublasscal(cubHandle, N, beta, d_p, 1));
            CHECK_CUBLAS(cublasAxpy(cubHandle, N, (T)1, d_z, 1, d_p, 1));
        }

        // q = A*pk
        CHECK_CUSPARSE(cusparseSpMV(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d_matA,
            dvec_p, &zero, dvec_q, dType, CUSPARSE_SPMV_CSR_ALG1, d_bufMV));

        // alpha = (rk*zk) / (pk*q)
        CHECK_CUBLAS(cublasdot(cubHandle, N, d_p, 1, d_q, 1, &pTq));
        alpha = rho / pTq;

        // x{k+1} = xk + alpha*pk
        CHECK_CUBLAS(cublasAxpy(cubHandle, N, alpha, d_p, 1, d_x, 1));

        // r{k+1} = rk - alpha*q 
        CHECK_CUBLAS(cublasAxpy(cubHandle, N, -alpha, d_q, 1, d_r, 1));
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(d_matA));
}

template class PCGJacobiSolver<float>;
template class PCGJacobiSolver<double>;