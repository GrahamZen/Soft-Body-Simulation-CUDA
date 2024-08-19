#include <simulation/solver/projective/pdSolver.h>
#include <simulation/solver/projective/pdUtil.cuh>
#include <simulation/simulationContext.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <utilities.cuh>

PdSolver::PdSolver(int threadsPerBlock, const SolverData& solverData) : FEMSolver(threadsPerBlock)
{
    cudaMalloc((void**)&solverData.dev_ExtForce, sizeof(glm::vec3) * solverData.numVerts);
    cudaMemset(solverData.dev_ExtForce, 0, sizeof(glm::vec3) * solverData.numVerts);
    cudaMalloc((void**)&solverData.V0, sizeof(float) * solverData.numTets);
    cudaMemset(solverData.V0, 0, sizeof(float) * solverData.numTets);
    cudaMalloc((void**)&solverData.inv_Dm, sizeof(glm::mat4) * solverData.numTets);

    int blocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    PdUtil::computeInvDmV0 << < blocks, threadsPerBlock >> > (solverData.V0, solverData.inv_Dm, solverData.numTets, solverData.X, solverData.Tet);
}

PdSolver::~PdSolver() {
    if (cholSpData) {
        free(cholSpData);
    }
    if (cholDnData) {
        free(cholDnData);
    }
    cudaFree(sn);
    cudaFree(b);
    cudaFree(masses);
    free(bHost);
}

__global__ void FillMatrixA(int* AIdx, float* tmpVal, float* d_A, int n, int ASize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int row = AIdx[idx] / ASize;
    int col = AIdx[idx] % ASize;
    atomicAdd(&d_A[row * ASize + col], tmpVal[idx]);
}

CholeskyDnData::~CholeskyDnData()
{
    cudaFree(d_info);
    cudaFree(d_A);
    cudaFree(d_work);
}

CholeskyDnData::CholeskyDnData(int threadsPerBlock, int* AIdx, float* tmpVal, int ASize, int len) {
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

CholeskySpData::~CholeskySpData()
{
    cusolverSpDestroyCsrcholInfo(d_info);
    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(cusolverHandle);
    cudaFree(buffer_gpu);
}

CholeskySpData::CholeskySpData(int threadsPerBlock, int* AIdx, float* tmpVal, int ASize, int len) {
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

    PdUtil::initAMatrix << < blocks, threadsPerBlock >> > (newIdx, ARowTmp, ACol, ASize, nnz);

    // transform ARow into csr format
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseXcoo2csr(handle, ARowTmp, nnz, ASize, ARow, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t descrA;
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
    cudaFree(ARowTmp);
    cudaFree(ARow);
    cudaFree(ACol);
    cudaFree(AVal);
}

void CholeskyDnData::Solve(float* d_b, int bSize, float* d_x) {
    cusolverDnXpotrs(cusolverHandle, params, CUBLAS_FILL_MODE_LOWER, bSize, 1, /* nrhs */
        cudaDataType::CUDA_R_32F, d_A, bSize,
        cudaDataType::CUDA_R_32F, d_b, bSize, d_info);
    cudaMemcpy(d_x, d_b, sizeof(float) * (bSize), cudaMemcpyDeviceToDevice);
}

void PdSolver::SolverPrepare(SolverData& solverData, SolverParams& solverParams)
{
    int vertBlocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    int tetBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    float dt = solverParams.dt;
    float const m_1_dt2 = solverParams.solverAttr.mass / (dt * dt);
    int len = solverData.numVerts * 3 + 48 * solverData.numTets;
    int ASize = 3 * solverData.numVerts;

    cudaMalloc((void**)&sn, sizeof(float) * ASize);
    cudaMalloc((void**)&b, sizeof(float) * ASize);
    cudaMalloc((void**)&masses, sizeof(float) * ASize);

    int* AIdx;
    cudaMalloc((void**)&AIdx, sizeof(int) * len);
    cudaMemset(AIdx, 0, sizeof(int) * len);

    float* tmpVal;
    cudaMalloc((void**)&tmpVal, sizeof(int) * len);
    cudaMemset(tmpVal, 0, sizeof(int) * len);

    PdUtil::computeSiTSi << < tetBlocks, threadsPerBlock >> > (AIdx, tmpVal, solverData.V0, solverData.inv_Dm, solverData.Tet, solverParams.solverAttr.stiffness_0, solverData.numTets, solverData.numVerts);
    PdUtil::setMDt_2 << < vertBlocks, threadsPerBlock >> > (AIdx, tmpVal, 48 * solverData.numTets, m_1_dt2, solverData.numVerts);

    bHost = (float*)malloc(sizeof(float) * ASize);
    int* AIdxHost = (int*)malloc(sizeof(int) * len);
    float* tmpValHost = (float*)malloc(sizeof(float) * len);

    cudaMemcpy(AIdxHost, AIdx, sizeof(int) * len, cudaMemcpyDeviceToHost);
    cudaMemcpy(tmpValHost, tmpVal, sizeof(float) * len, cudaMemcpyDeviceToHost);

    std::vector<Eigen::Triplet<float>> A_triplets;

    for (auto i = 0; i < len; ++i)
    {
        A_triplets.push_back({ AIdxHost[i] / ASize, AIdxHost[i] % ASize, tmpValHost[i] });
    }
    Eigen::SparseMatrix<float> A(ASize, ASize);

    A.setFromTriplets(A_triplets.begin(), A_triplets.end());
    cholesky_decomposition_.compute(A);

    cholDnData = new CholeskyDnData(threadsPerBlock, AIdx, tmpVal, ASize, len);
    //cholSpData = new CholeskySpData(threadsPerBlock, AIdx, tmpVal, ASize, len);

    cudaFree(AIdx);
    cudaFree(tmpVal);
}


void PdSolver::SolverStep(SolverData& solverData, SolverParams& solverParams)
{

    float dt = solverParams.dt;
    float const dtInv = 1.0f / dt;
    float const dt2 = dt * dt;
    float const dt2_m_1 = dt2 / solverParams.solverAttr.mass;
    float const m_1_dt2 = 1.f / dt2_m_1;

    int vertBlocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    int tetBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;

    glm::vec3 gravity{ 0.0f, -solverParams.gravity * solverParams.solverAttr.mass, 0.0f };
    thrust::device_ptr<glm::vec3> dev_ptr(solverData.dev_ExtForce);
    thrust::fill(thrust::device, dev_ptr, dev_ptr + solverData.numVerts, gravity);
    PdUtil::computeSn << < vertBlocks, threadsPerBlock >> > (sn, dt, dt2_m_1, solverData.X, solverData.V, solverData.dev_ExtForce, masses, m_1_dt2, solverData.numVerts);
    for (int i = 0; i < solverParams.numIterations; i++)
    {
        cudaMemset(b, 0, sizeof(float) * solverData.numVerts * 3);
        PdUtil::computeLocal << < tetBlocks, threadsPerBlock >> > (solverData.V0, solverParams.solverAttr.stiffness_0, b, solverData.inv_Dm, sn, solverData.Tet, solverData.numTets);
        PdUtil::addM_h2Sn << < vertBlocks, threadsPerBlock >> > (b, masses, solverData.numVerts);

        if (useEigen)
        {
            cudaMemcpy(bHost, b, sizeof(float) * (solverData.numVerts * 3), cudaMemcpyDeviceToHost);
            Eigen::VectorXf bh = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(bHost, solverData.numVerts * 3);
            Eigen::VectorXf res = cholesky_decomposition_.solve(bh);
            cudaMemcpy(sn, res.data(), sizeof(float) * (solverData.numVerts * 3), cudaMemcpyHostToDevice);
        }
        else
        {
            cholDnData->Solve(b, solverData.numVerts * 3, sn);
        }
    }

    PdUtil::updateVelPos << < vertBlocks, threadsPerBlock >> > (sn, dtInv, solverData.XTilt, solverData.V, solverData.numVerts);
}


void PdSolver::Update(SolverData& solverData, SolverParams& solverParams)
{
    AddExternal << <(solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (solverData.V, solverData.numVerts, solverParams.solverAttr.jump, solverParams.solverAttr.mass, solverParams.extForce.jump);
    if (!solverReady)
    {
        SolverPrepare(solverData, solverParams);
        solverReady = true;
    }
    SolverStep(solverData, solverParams);
}

void CholeskySpData::Solve(float* d_b, int bSize, float* d_x)
{
    cusolverSpScsrcholSolve(cusolverHandle, bSize, d_b, d_x, d_info, buffer_gpu);
}