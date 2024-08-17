#include <simulation/solver/projective/pdSolver.h>
#include <simulation/solver/projective/pdUtil.cuh>
#include <simulation/simulationContext.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <utilities.cuh>

PdSolver::PdSolver(SimulationCUDAContext* context, const SolverData& solverData) : FEMSolver(context)
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
    cudaFree(sn);
    cudaFree(b);
    cudaFree(masses);
    cudaFree(d_A);
    cudaFree(d_info);
    cudaFree(d_work);
    free(bHost);
}


void PdSolver::SolverPrepare(SolverData& solverData, SolverAttribute& attrib)
{
    int vertBlocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    int tetBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    float dt = mcrpSimContext->GetDt();
    float const m_1_dt2 = attrib.mass / (dt * dt);
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

    PdUtil::computeSiTSi << < tetBlocks, threadsPerBlock >> > (AIdx, tmpVal, solverData.V0, solverData.inv_Dm, solverData.Tet, attrib.stiffness_0, solverData.numTets, solverData.numVerts);
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

    free(AIdxHost);
    free(tmpValHost);

    Eigen::MatrixXf ADense = Eigen::MatrixXf(A);

    cusolverDnCreate(&cusolverHandle);
    cusolverDnCreateParams(&params);

    // Matrix dimension and leading dimension
    int n = ASize;  // Matrix size (assuming square matrix)
    int lda = ASize;  // Leading dimension of A
    int info = 0;
    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void* h_work = nullptr;              /* host workspace */
    // Allocate memory for dense matrix A
    cudaMalloc(&d_A, sizeof(float) * n * n);
    cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int));

    // Copy your matrix data from host to device
    // Assuming h_A is the host matrix with size n x n
    cudaMemcpy(d_A, ADense.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);

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
    cudaFree(AIdx);
    cudaFree(tmpVal);
}


void PdSolver::SolverStep(SolverData& solverData, SolverAttribute& attrib)
{

    float dt = mcrpSimContext->GetDt();
    float const dtInv = 1.0f / dt;
    float const dt2 = dt * dt;
    float const dt2_m_1 = dt2 / attrib.mass;
    float const m_1_dt2 = 1.f / dt2_m_1;


    int vertBlocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    int tetBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;

    glm::vec3 gravity{ 0.0f, -mcrpSimContext->GetGravity() * attrib.mass, 0.0f };
    thrust::device_ptr<glm::vec3> dev_ptr(solverData.dev_ExtForce);
    thrust::fill(thrust::device, dev_ptr, dev_ptr + solverData.numVerts, gravity);
    //computeSn << < vertBlocks, threadsPerBlock >> > (sn, dt, dt2_m_1, solverData.X, solverData.V, thrust::raw_pointer_cast(solverData.dev_ExtForce.data()), masses, m_1_dt2, solverData.numVerts);
    PdUtil::computeSn << < vertBlocks, threadsPerBlock >> > (sn, dt, dt2_m_1, solverData.X, solverData.V, solverData.dev_ExtForce, masses, m_1_dt2, solverData.numVerts);
    for (int i = 0; i < mcrpSimContext->GetNumIterations(); i++)
    {
        cudaMemset(b, 0, sizeof(float) * solverData.numVerts * 3);
        PdUtil::computeLocal << < tetBlocks, threadsPerBlock >> > (solverData.V0, attrib.stiffness_0, b, solverData.inv_Dm, sn, solverData.Tet, solverData.numTets);
        PdUtil::addM_h2Sn << < vertBlocks, threadsPerBlock >> > (b, masses, solverData.numVerts);

        if (mcrpSimContext->IsEigenGlobalSolver())
        {
            cudaMemcpy(bHost, b, sizeof(float) * (solverData.numVerts * 3), cudaMemcpyDeviceToHost);
            Eigen::VectorXf bh = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(bHost, solverData.numVerts * 3);
            Eigen::VectorXf res = cholesky_decomposition_.solve(bh);
            cudaMemcpy(sn, res.data(), sizeof(float) * (solverData.numVerts * 3), cudaMemcpyHostToDevice);
        }
        else
        {
            cusolverDnXpotrs(cusolverHandle, params, CUBLAS_FILL_MODE_LOWER, solverData.numVerts * 3, 1, /* nrhs */
                cudaDataType::CUDA_R_32F, d_A, solverData.numVerts * 3,
                cudaDataType::CUDA_R_32F, b, solverData.numVerts * 3, d_info);
            cudaMemcpy(sn, b, sizeof(float) * (solverData.numVerts * 3), cudaMemcpyDeviceToDevice);
        }
    }

    PdUtil::updateVelPos << < vertBlocks, threadsPerBlock >> > (sn, dtInv, solverData.XTilt, solverData.V, solverData.numVerts);
}


void PdSolver::Update(SolverData& solverData, SolverAttribute& attrib)
{
    AddExternal << <(solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (solverData.V, solverData.numVerts, attrib.jump, attrib.mass, mcrpSimContext->GetExtForce().jump);
    if (!solverReady)
    {
        SolverPrepare(solverData, attrib);
        solverReady = true;
    }
    SolverStep(solverData, attrib);
}