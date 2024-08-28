#include <simulation/solver/linear/cholesky.h>
#include <thrust/execution_policy.h>
#include <linear/linearUtils.cuh>

template<typename T>
__global__ void FillMatrixA(int* AIdx, T* tmpVal, T* d_A, int n, int ASize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int row = AIdx[idx] / ASize;
    int col = AIdx[idx] % ASize;
    atomicAdd(&d_A[row * ASize + col], tmpVal[idx]);
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
CholeskyDnLinearSolver<T>::CholeskyDnLinearSolver(int threadsPerBlock, int* AIdx, T* tmpVal, int ASize, int len) {
    cudaMalloc(&d_predecomposedA, sizeof(T) * ASize * ASize);
    FillMatrixA << < (len + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (AIdx, tmpVal, d_predecomposedA, len, ASize);
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
        cusolverHandle, params, CUBLAS_FILL_MODE_LOWER, n, dataType, d_predecomposedA, lda,
        dataType, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);

    cudaMalloc(reinterpret_cast<void**>(&d_work), workspaceInBytesOnDevice);
    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void*>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    cusolverDnXpotrf(cusolverHandle, params, CUBLAS_FILL_MODE_LOWER, n, dataType,
        d_predecomposedA, lda, dataType, d_work, workspaceInBytesOnDevice,
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

template<>
CholeskySpLinearSolver<double>::CholeskySpLinearSolver(int threadsPerBlock, int* ARow, int* ACol, double* AVal, int ASize, int len) {
    sort_coo(ASize, len, AVal, ARow, ACol);
    int nnz = len;
    // transform ARow into csr format
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseXcoo2csr(handle, ARow, nnz, ASize, ARow, CUSPARSE_INDEX_BASE_ZERO);

    cusolverSpCreate(&cusolverHandle);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    ComputeAMD(cusolverHandle, ASize, nnz, ARow, ACol, AVal);
    size_t cholSize = 0;
    size_t internalSize = 0;
    cusolverSpCreateCsrcholInfo(&d_info);
    cusolverSpXcsrcholAnalysis(cusolverHandle, ASize, nnz, descrA, ARow, ACol, d_info);
    cusolverSpDcsrcholBufferInfo(cusolverHandle, ASize, nnz, descrA, AVal, ARow, ACol, d_info, &internalSize, &cholSize);
    cudaMalloc((void**)&buffer_gpu, sizeof(char) * cholSize);
    cudaMalloc((void**)&dev_b_permuted, sizeof(double) * ASize);
    cudaMalloc((void**)&dev_x_permuted, sizeof(double) * ASize);
    cusolverSpDcsrcholFactor(cusolverHandle, ASize, nnz, descrA, AVal, ARow, ACol, d_info, buffer_gpu);
}

template<> CholeskySpLinearSolver<float>::CholeskySpLinearSolver(int threadsPerBlock, int* ARow, int* ACol, float* AVal, int ASize, int len) {
    sort_coo(ASize, len, AVal, ARow, ACol);
    int nnz = len;
    // transform ARow into csr format
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseXcoo2csr(handle, ARow, nnz, ASize, ARow, CUSPARSE_INDEX_BASE_ZERO);

    cusolverSpCreate(&cusolverHandle);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    ComputeAMD(cusolverHandle, ASize, nnz, ARow, ACol, AVal);
    size_t cholSize = 0;
    size_t internalSize = 0;
    cusolverSpCreateCsrcholInfo(&d_info);
    cusolverSpXcsrcholAnalysis(cusolverHandle, ASize, nnz, descrA, ARow, ACol, d_info);
    cusolverSpScsrcholBufferInfo(cusolverHandle, ASize, nnz, descrA, AVal, ARow, ACol, d_info, &internalSize, &cholSize);
    cudaMalloc((void**)&buffer_gpu, sizeof(char) * cholSize);
    cudaMalloc((void**)&dev_b_permuted, sizeof(float) * ASize);
    cudaMalloc((void**)&dev_x_permuted, sizeof(float) * ASize);
    cusolverSpScsrcholFactor(cusolverHandle, ASize, nnz, descrA, AVal, ARow, ACol, d_info, buffer_gpu);
}

template<typename T>
void CholeskyDnLinearSolver<T>::Solve(int N, T* d_b, T* d_x, T* d_A, int nz, int* d_rowIdx, int* d_colIdx, T* d_guess) {
    cusolverDnXpotrs(cusolverHandle, params, CUBLAS_FILL_MODE_LOWER, N, 1, /* nrhs */
        dataType, d_predecomposedA, N,
        dataType, d_b, N, d_info);
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

template<> void CholeskySpLinearSolver<float>::Solve(int N, float* d_b, float* d_x, float* d_A, int nz, int* d_rowIdx, int* d_colIdx, float* d_guess)
{
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    permuteVector << <blocks, threadsPerBlock >> > (d_b, dev_b_permuted, d_p, N);
    cusolverSpScsrcholSolve(cusolverHandle, N, dev_b_permuted, dev_x_permuted, d_info, buffer_gpu);
    permuteVectorInv << <blocks, threadsPerBlock >> > (dev_x_permuted, d_x, d_p, N);
}

template<> void CholeskySpLinearSolver<double>::Solve(int N, double* d_b, double* d_x, double* d_A, int nz, int* d_rowIdx, int* d_colIdx, double* d_guess)
{
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    permuteVector << <blocks, threadsPerBlock >> > (d_b, dev_b_permuted, d_p, N);
    cusolverSpDcsrcholSolve(cusolverHandle, N, dev_b_permuted, dev_x_permuted, d_info, buffer_gpu);
    permuteVectorInv << <blocks, threadsPerBlock >> > (dev_x_permuted, d_x, d_p, N);
}

template CholeskySpLinearSolver<double>;
template CholeskySpLinearSolver<float>;