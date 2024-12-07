#include <simulation/solver/linear/cholesky.h>
#include <linear/cuUtils.cuh>
#include <stdexcept>

template<typename T>
__global__ void FillMatrixA(int* AIdx, T* AVal, T* d_A, int n, int ASize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int row = AIdx[idx] / ASize;
    int col = AIdx[idx] % ASize;
    atomicAdd(&d_A[row * ASize + col], AVal[idx]);
}

__global__ void initAMatrix(int* idx, int* row, int* col, int rowLen, int totalNumber)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < totalNumber)
    {
        row[index] = idx[index] / rowLen;
        col[index] = idx[index] % rowLen;
    }
}

template<typename T>
CholeskyDnLinearSolver<T>::~CholeskyDnLinearSolver()
{
    cudaFree(d_info);
    cudaFree(d_predecomposedA);
    cudaFree(d_work);
}

template<typename T>
CholeskyDnLinearSolver<T>::CholeskyDnLinearSolver(int threadsPerBlock, int* AIdx, T* AVal, int ASize, int len) {
    cudaMalloc(&d_predecomposedA, sizeof(T) * ASize * ASize);
    FillMatrixA << < (len + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (AIdx, AVal, d_predecomposedA, len, ASize);
    cusolverDnCreate(&cusolverHandle);
    cusolverDnCreateParams(&params);

    // Matrix dimension and leading dimension
    int n = ASize;
    int lda = n;  // Leading dimension of A
    int info = 0;
    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void* h_work = nullptr;              /* host workspace */
    // Allocate memory for dense matrix A
    cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int));

    // Copy your matrix data from host to device
    // Assuming h_A is the host matrix with size n x n

    cusolverDnXpotrf_bufferSize(
        cusolverHandle, params, CUBLAS_FILL_MODE_LOWER, n, dType, d_predecomposedA, lda,
        dType, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);

    cudaMalloc(reinterpret_cast<void**>(&d_work), workspaceInBytesOnDevice);
    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void*>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    cusolverDnXpotrf(cusolverHandle, params, CUBLAS_FILL_MODE_LOWER, n, dType,
        d_predecomposedA, lda, dType, d_work, workspaceInBytesOnDevice,
        h_work, workspaceInBytesOnHost, d_info);
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    std::printf("after Xpotrf: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    free(h_work);
}

template<typename T>
CholeskySpLinearSolver<T>::~CholeskySpLinearSolver()
{
    cusolverSpDestroyCsrcholInfo(d_info);
    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(cusolverHandle);
    cudaFree(buffer_gpu);
    cudaFree(dev_x_permuted);
    cudaFree(dev_b_permuted);
    cudaFree(d_rowPtrA);
}

template<typename T>
void CholeskySpLinearSolver<T>::ComputeAMD(cusolverSpHandle_t handle, int rowsA, int nnzA, int* dev_csrRowPtrA, int* dev_csrColIndA, T* dev_csrValA) {
    std::vector<int> h_Q(rowsA);
    std::vector<int> h_csrRowPtrB(rowsA + 1);
    std::vector<int> h_csrColIndB(nnzA);
    std::vector<T> h_csrValB(nnzA);
    std::vector<int> h_mapBfromA(nnzA);

    std::vector<int> h_csrRowPtrA(rowsA + 1);
    std::vector<int> h_csrColIndA(nnzA);
    std::vector<T> h_csrValA(nnzA);

    cudaMemcpy(h_csrRowPtrA.data(), dev_csrRowPtrA, sizeof(int) * (rowsA + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrColIndA.data(), dev_csrColIndA, sizeof(int) * nnzA, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrValA.data(), dev_csrValA, sizeof(T) * nnzA, cudaMemcpyDeviceToHost);

    cusolverSpXcsrsymamdHost(
        handle, rowsA, nnzA,
        descrA, h_csrRowPtrA.data(), h_csrColIndA.data(),
        h_Q.data());

    // B = Q*A*Q^T
    memcpy(h_csrRowPtrB.data(), h_csrRowPtrA.data(), sizeof(int) * (rowsA + 1));
    memcpy(h_csrColIndB.data(), h_csrColIndA.data(), sizeof(int) * nnzA);

    size_t size_perm;
    cusolverSpXcsrperm_bufferSizeHost(
        handle, rowsA, rowsA, nnzA,
        descrA, h_csrRowPtrB.data(), h_csrColIndB.data(),
        h_Q.data(), h_Q.data(),
        &size_perm);
    void* buffer_cpu = nullptr;
    buffer_cpu = (void*)malloc(sizeof(char) * size_perm);
    assert(NULL != buffer_cpu);

    // h_mapBfromA.data() = Identity
    for (int j = 0; j < nnzA; j++)
    {
        h_mapBfromA.data()[j] = j;
    }
    cusolverSpXcsrpermHost(
        handle, rowsA, rowsA, nnzA,
        descrA, h_csrRowPtrB.data(), h_csrColIndB.data(),
        h_Q.data(), h_Q.data(),
        h_mapBfromA.data(),
        buffer_cpu);

    // B = A( mapBfromA )
    for (int j = 0; j < nnzA; j++)
    {
        h_csrValB[j] = h_csrValA[h_mapBfromA[j]];
    }

    cudaMemcpy(dev_csrRowPtrA, h_csrRowPtrB.data(), sizeof(int) * (rowsA + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_csrColIndA, h_csrColIndB.data(), sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_csrValA, h_csrValB.data(), sizeof(T) * nnzA, cudaMemcpyHostToDevice);
    cudaMalloc(&d_p, sizeof(int) * rowsA);
    cudaMemcpy(d_p, h_Q.data(), sizeof(int) * rowsA, cudaMemcpyHostToDevice);
    free(buffer_cpu);
}

template<typename T>
CholeskySpLinearSolver<T>::CholeskySpLinearSolver(int threadsPerBlock, int* rowIdx, int* colIdx, T* A, int ASize, int len) {
    sort_coo(ASize, len, A, rowIdx, colIdx, d_A, d_rowIdx, d_colIdx);
    int nnz = len;
    cudaMalloc((void**)&d_rowPtrA, sizeof(int) * (ASize + 1));
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseXcoo2csr(handle, d_rowIdx, nnz, ASize, d_rowPtrA, CUSPARSE_INDEX_BASE_ZERO);

    cusolverSpCreate(&cusolverHandle);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    ComputeAMD(cusolverHandle, ASize, nnz, d_rowPtrA, d_colIdx, d_A);
    size_t cholSize = 0;
    size_t internalSize = 0;
    cusolverSpCreateCsrcholInfo(&d_info);
    cusolverSpXcsrcholAnalysis(cusolverHandle, ASize, nnz, descrA, d_rowPtrA, d_colIdx, d_info);
    cusolverSpcsrcholBufferInfo(cusolverHandle, ASize, nnz, descrA, d_A, d_rowPtrA, d_colIdx, d_info, &internalSize, &cholSize);
    cudaMalloc((void**)&buffer_gpu, sizeof(char) * cholSize);
    cudaMalloc((void**)&dev_b_permuted, sizeof(T) * ASize);
    cudaMalloc((void**)&dev_x_permuted, sizeof(T) * ASize);
    cusolverSpcsrcholFactor(cusolverHandle, ASize, nnz, descrA, d_A, d_rowPtrA, d_colIdx, d_info, buffer_gpu);
}

template<typename T>
void CholeskyDnLinearSolver<T>::Solve(int N, T* d_b, T* d_x, T* d_A, int nz, int* d_rowIdx, int* d_colIdx, T* d_guess) {
    cusolverDnXpotrs(cusolverHandle, params, CUBLAS_FILL_MODE_LOWER, N, 1, /* nrhs */
        dType, d_predecomposedA, N,
        dType, d_b, N, d_info);
    cudaMemcpy(d_x, d_b, sizeof(T) * (N), cudaMemcpyDeviceToDevice);
}

template<typename T>
__global__ void permuteVector(const T* b, T* b_permuted, const int* p, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b_permuted[idx] = b[p[idx]];
    }
}

template<typename T>
__global__ void permuteVectorInv(const T* x_permuted, T* x, const int* p, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[p[idx]] = x_permuted[idx];
    }
}

template<typename T>
void CholeskySpLinearSolver<T>::Solve(int N, T* d_b, T* d_x, T* d_A, int nz, int* d_rowIdx, int* d_colIdx, T* d_guess)
{
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    permuteVector << <blocks, threadsPerBlock >> > (d_b, dev_b_permuted, d_p, N);
    cusolverSpcsrcholSolve(cusolverHandle, N, dev_b_permuted, dev_x_permuted, d_info, buffer_gpu);
    permuteVectorInv << <blocks, threadsPerBlock >> > (dev_x_permuted, d_x, d_p, N);
}

template CholeskySpLinearSolver<double>;
template CholeskySpLinearSolver<float>;