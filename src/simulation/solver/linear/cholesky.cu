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

CholeskyDnLinearSolver::~CholeskyDnLinearSolver()
{
    cudaFree(d_info);
    cudaFree(d_predecomposedA);
    cudaFree(d_work);
}

CholeskyDnLinearSolver::CholeskyDnLinearSolver(int threadsPerBlock, int* AIdx, float* tmpVal, int ASize, int len) {
    cudaMalloc(&d_predecomposedA, sizeof(float) * ASize * ASize);
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
        cusolverHandle, params, CUBLAS_FILL_MODE_LOWER, n, cudaDataType::CUDA_R_32F, d_predecomposedA, lda,
        cudaDataType::CUDA_R_32F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);

    cudaMalloc(reinterpret_cast<void**>(&d_work), workspaceInBytesOnDevice);
    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void*>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    cusolverDnXpotrf(cusolverHandle, params, CUBLAS_FILL_MODE_LOWER, n, cudaDataType::CUDA_R_32F,
        d_predecomposedA, lda, cudaDataType::CUDA_R_32F, d_work, workspaceInBytesOnDevice,
        h_work, workspaceInBytesOnHost, d_info);
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    std::printf("after Xpotrf: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    free(h_work);
}

CholeskySpLinearSolver::~CholeskySpLinearSolver()
{
    cusolverSpDestroyCsrcholInfo(d_info);
    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(cusolverHandle);
    cudaFree(buffer_gpu);
}

CholeskySpLinearSolver::CholeskySpLinearSolver(int threadsPerBlock, int* AIdx, float* tmpVal, int ASize, int len) {
    int* newIdx;
    float* newVal;

    cudaMalloc((void**)&newIdx, sizeof(int) * len);
    cudaMalloc((void**)&newVal, sizeof(float) * len);

    thrust::sort_by_key(thrust::device, AIdx, AIdx + len, tmpVal);
    thrust::pair<int*, float*> newEnd = thrust::reduce_by_key(thrust::device, AIdx, AIdx + len, tmpVal, newIdx, newVal);

    int* ARow, * ACol;
    float* AVal;

    int nnz = newEnd.first - newIdx;

    cudaMalloc((void**)&ARow, sizeof(int) * nnz);
    cudaMemset(ARow, 0, sizeof(int) * nnz);

    cudaMalloc((void**)&ACol, sizeof(int) * nnz);
    cudaMemset(ACol, 0, sizeof(int) * nnz);

    cudaMalloc((void**)&AVal, sizeof(float) * nnz);
    cudaMemcpy(AVal, newVal, sizeof(float) * nnz, cudaMemcpyDeviceToDevice);

    int blocks = (nnz + threadsPerBlock - 1) / threadsPerBlock;

    initAMatrix << < blocks, threadsPerBlock >> > (newIdx, ARow, ACol, ASize, nnz);

    // transform ARow into csr format
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseXcoo2csr(handle, ARow, nnz, ASize, ARow, CUSPARSE_INDEX_BASE_ZERO);

    cusolverSpCreate(&cusolverHandle);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    size_t cholSize = 0;
    size_t internalSize = 0;
    cusolverSpCreateCsrcholInfo(&d_info);
    cusolverSpXcsrcholAnalysis(cusolverHandle, ASize, nnz, descrA, ARow, ACol, d_info);
    cusolverSpScsrcholBufferInfo(cusolverHandle, ASize, nnz, descrA, AVal, ARow, ACol, d_info, &internalSize, &cholSize);
    cudaMalloc(&buffer_gpu, sizeof(char) * cholSize);
    cusolverSpScsrcholFactor(cusolverHandle, ASize, nnz, descrA, AVal, ARow, ACol, d_info, buffer_gpu);

    cudaFree(newIdx);
    cudaFree(newVal);
    cudaFree(ARow);
    cudaFree(ACol);
    cudaFree(AVal);
}
void CholeskyDnLinearSolver::Solve(int N, float* d_b, float* d_x, float* d_A, int nz, int* d_rowIdx, int* d_colIdx, float* d_guess) {
    cusolverDnXpotrs(cusolverHandle, params, CUBLAS_FILL_MODE_LOWER, N, 1, /* nrhs */
        cudaDataType::CUDA_R_32F, d_predecomposedA, N,
        cudaDataType::CUDA_R_32F, d_b, N, d_info);
    cudaMemcpy(d_x, d_b, sizeof(float) * (N), cudaMemcpyDeviceToDevice);
}

void CholeskySpLinearSolver::Solve(int N, float* d_b, float* d_x, float* d_A, int nz, int* d_rowIdx, int* d_colIdx, float* d_guess)
{
    cusolverSpScsrcholSolve(cusolverHandle, N, d_b, d_x, d_info, buffer_gpu);
}
