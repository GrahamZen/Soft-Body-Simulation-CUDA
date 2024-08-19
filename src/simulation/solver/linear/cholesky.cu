#include <simulation/solver/linear/cholesky.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

__global__ void FillMatrixA(int* AIdx, float* tmpVal, float* d_A, int n, int ASize) {
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

CholeskyDnlinearSolver::~CholeskyDnlinearSolver()
{
    cudaFree(d_info);
    cudaFree(d_A);
    cudaFree(d_work);
}

CholeskyDnlinearSolver::CholeskyDnlinearSolver(int threadsPerBlock, int* AIdx, float* tmpVal, int ASize, int len) {
    cudaMalloc(&d_A, sizeof(float) * ASize * ASize);
    FillMatrixA << < (len + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (AIdx, tmpVal, d_A, len, ASize);
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
        cusolverHandle, params, CUBLAS_FILL_MODE_LOWER, n, cudaDataType::CUDA_R_32F, d_A, lda,
        cudaDataType::CUDA_R_32F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);

    cudaMalloc(reinterpret_cast<void**>(&d_work), workspaceInBytesOnDevice);
    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void*>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    cusolverDnXpotrf(cusolverHandle, params, CUBLAS_FILL_MODE_LOWER, n, cudaDataType::CUDA_R_32F,
        d_A, lda, cudaDataType::CUDA_R_32F, d_work, workspaceInBytesOnDevice,
        h_work, workspaceInBytesOnHost, d_info);
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    std::printf("after Xpotrf: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    free(h_work);
}

CholeskySplinearSolver::~CholeskySplinearSolver()
{
    cusolverSpDestroyCsrcholInfo(d_info);
    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(cusolverHandle);
    cudaFree(buffer_gpu);
}

CholeskySplinearSolver::CholeskySplinearSolver(int threadsPerBlock, int* AIdx, float* tmpVal, int ASize, int len) {
    int* newIdx;
    float* newVal;

    cudaMalloc((void**)&newIdx, sizeof(int) * len);
    cudaMalloc((void**)&newVal, sizeof(float) * len);

    thrust::sort_by_key(thrust::device, AIdx, AIdx + len, tmpVal);


    thrust::pair<int*, float*> newEnd = thrust::reduce_by_key(thrust::device, AIdx, AIdx + len, tmpVal, newIdx, newVal);

    int* ARow;
    int* ACol;
    float* AVal;

    nnz = newEnd.first - newIdx;
    std::cout << nnz << std::endl;

    cudaMalloc((void**)&ARow, sizeof(int) * nnz);
    cudaMemset(ARow, 0, sizeof(int) * nnz);

    cudaMalloc((void**)&ACol, sizeof(int) * nnz);
    cudaMemset(ACol, 0, sizeof(int) * nnz);

    cudaMalloc((void**)&AVal, sizeof(float) * nnz);
    cudaMemcpy(AVal, newVal, sizeof(float) * nnz, cudaMemcpyDeviceToDevice);

    int* ARowTmp;
    cudaMalloc((void**)&ARowTmp, sizeof(int) * nnz);
    cudaMemset(ARowTmp, 0, sizeof(int) * nnz);

    int blocks = (nnz + threadsPerBlock - 1) / threadsPerBlock;

    initAMatrix << < blocks, threadsPerBlock >> > (newIdx, ARowTmp, ACol, ASize, nnz);

    // transform ARow into csr format
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseXcoo2csr(handle, ARowTmp, nnz, ASize, ARow, CUSPARSE_INDEX_BASE_ZERO);

    cusolverSpCreate(&cusolverHandle);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    size_t cholSize = 0;
    size_t internalSize = 0;
    cusolverSpCreateCsrcholInfo(&d_info);
    ApproximateMinimumDegree(ASize, ARow, ACol, AVal);
    cusolverSpXcsrcholAnalysis(cusolverHandle, ASize, nnz, descrA, ARow, ACol, d_info);
    cusolverSpScsrcholBufferInfo(cusolverHandle, ASize, nnz, descrA, AVal, ARow, ACol, d_info, &internalSize, &cholSize);
    cudaMalloc(&buffer_gpu, sizeof(char) * cholSize);
    cusolverSpScsrcholFactor(cusolverHandle, ASize, nnz, descrA, AVal, ARow, ACol, d_info, buffer_gpu);

    cudaFree(newIdx);
    cudaFree(newVal);
    cudaFree(ARowTmp);
    cudaFree(ARow);
    cudaFree(ACol);
    cudaFree(AVal);
}

void CholeskyDnlinearSolver::Solve(float* d_b, int bSize, float* d_x) {
    cusolverDnXpotrs(cusolverHandle, params, CUBLAS_FILL_MODE_LOWER, bSize, 1, /* nrhs */
        cudaDataType::CUDA_R_32F, d_A, bSize,
        cudaDataType::CUDA_R_32F, d_b, bSize, d_info);
    cudaMemcpy(d_x, d_b, sizeof(float) * (bSize), cudaMemcpyDeviceToDevice);
}

void CholeskySplinearSolver::Solve(float* d_b, int bSize, float* d_x)
{
    cusolverSpScsrcholSolve(cusolverHandle, bSize, d_b, d_x, d_info, buffer_gpu);
}

void CholeskySplinearSolver::ApproximateMinimumDegree(int ASize, int* dev_ARow, int* dev_ACol, float* dev_AVal)
{
    size_t size_perm = 0;
    void* buffer_cpu = nullptr;
    int* h_csrRowPtrA = new int[ASize + 1];
    int* h_csrColIndA = new int[nnz];
    float* h_csrValA = new float[nnz];

    cudaMemcpy(h_csrRowPtrA, dev_ARow, sizeof(int) * (ASize + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrColIndA, dev_ACol, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrValA, dev_AVal, sizeof(float) * nnz, cudaMemcpyDeviceToHost);

    int colsA = ASize, rowsA = ASize, nnzA = nnz;

    int*  h_Q = (int*)malloc(sizeof(int) * colsA);
    int*  h_csrRowPtrB = (int*)malloc(sizeof(int) * (rowsA + 1));
    int*  h_csrColIndB = (int*)malloc(sizeof(int) * nnzA);
    double*  h_csrValB = (double*)malloc(sizeof(double) * nnzA);
    int*  h_mapBfromA = (int*)malloc(sizeof(int) * nnzA);

    assert(NULL != h_Q);
    assert(NULL != h_csrRowPtrB);
    assert(NULL != h_csrColIndB);
    assert(NULL != h_csrValB);
    assert(NULL != h_mapBfromA);

    (cusolverSpXcsrsymamdHost(
        cusolverHandle, rowsA, nnzA,
        descrA, h_csrRowPtrA, h_csrColIndA,
        h_Q));

    // B = Q*A*Q^T
    memcpy(h_csrRowPtrB, h_csrRowPtrA, sizeof(int) * (rowsA + 1));
    memcpy(h_csrColIndB, h_csrColIndA, sizeof(int) * nnzA);

    (cusolverSpXcsrperm_bufferSizeHost(
        cusolverHandle, rowsA, colsA, nnzA,
        descrA, h_csrRowPtrB, h_csrColIndB,
        h_Q, h_Q,
        &size_perm));

    if (buffer_cpu)
    {
        free(buffer_cpu);
    }
    buffer_cpu = (void*)malloc(sizeof(char) * size_perm);
    assert(NULL != buffer_cpu);

    // h_mapBfromA = Identity
    for (int j = 0; j < nnzA; j++)
    {
        h_mapBfromA[j] = j;
    }
    (cusolverSpXcsrpermHost(
        cusolverHandle, rowsA, colsA, nnzA,
        descrA, h_csrRowPtrB, h_csrColIndB,
        h_Q, h_Q,
        h_mapBfromA,
        buffer_cpu));

    // B = A( mapBfromA )
    for (int j = 0; j < nnzA; j++)
    {
        h_csrValB[j] = h_csrValA[h_mapBfromA[j]];
    }

    // A := B
    memcpy(h_csrRowPtrA, h_csrRowPtrB, sizeof(int) * (rowsA + 1));
    memcpy(h_csrColIndA, h_csrColIndB, sizeof(int) * nnzA);
    memcpy(h_csrValA, h_csrValB, sizeof(double) * nnzA);
    
    if (buffer_cpu)
    {
        free(buffer_cpu);
    }
}
