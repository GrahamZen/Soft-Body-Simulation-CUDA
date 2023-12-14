#include <pdSolver.h>
#include <utilities.cuh>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>
#include <simulationContext.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

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

    computeSiTSi << < tetBlocks, threadsPerBlock >> > (AIdx, tmpVal, V0, solverData.inv_Dm, solverData.Tet, attrib.stiffness_0, solverData.numTets, solverData.numVerts);
    setMDt_2 << < vertBlocks, threadsPerBlock >> > (AIdx, tmpVal, 48 * solverData.numTets, m_1_dt2, solverData.numVerts);

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

    int* newIdx;
    float* newVal;

    cudaMalloc((void**)&newIdx, sizeof(int) * len);
    cudaMalloc((void**)&newVal, sizeof(float) * len);

    thrust::sort_by_key(thrust::device, AIdx, AIdx + len, tmpVal);


    thrust::pair<int*, float*> newEnd = thrust::reduce_by_key(thrust::device, AIdx, AIdx + len, tmpVal, newIdx, newVal);

    int* ARow;
    int* ACol;
    float* AVal;

    nnzNumber = newEnd.first - newIdx;
    std::cout << nnzNumber << std::endl;

    cudaMalloc((void**)&ARow, sizeof(int) * nnzNumber);
    cudaMemset(ARow, 0, sizeof(int) * nnzNumber);

    cudaMalloc((void**)&ACol, sizeof(int) * nnzNumber);
    cudaMemset(ACol, 0, sizeof(int) * nnzNumber);

    cudaMalloc((void**)&AVal, sizeof(float) * nnzNumber);
    cudaMemcpy(AVal, newVal, sizeof(float) * nnzNumber, cudaMemcpyDeviceToDevice);

    int* ARowTmp;
    cudaMalloc((void**)&ARowTmp, sizeof(int) * nnzNumber);
    cudaMemset(ARowTmp, 0, sizeof(int) * nnzNumber);

    int blocks = (nnzNumber + threadsPerBlock - 1) / threadsPerBlock;

    initAMatrix << < blocks, threadsPerBlock >> > (newIdx, ARowTmp, ACol, ASize, nnzNumber);

    // transform ARow into csr format
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseXcoo2csr(handle, ARowTmp, nnzNumber, ASize, ARow, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t descrA;
    cusolverSpCreate(&cusolverHandle);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    size_t cholSize = 0;
    size_t internalSize = 0;
    cusolverSpCreateCsrcholInfo(&d_info);
    cusolverSpXcsrcholAnalysis(cusolverHandle, ASize, nnzNumber, descrA, ARow, ACol, d_info);
    cusolverSpScsrcholBufferInfo(cusolverHandle, ASize, nnzNumber, descrA, AVal, ARow, ACol, d_info, &internalSize, &cholSize);
    cudaMalloc(&buffer_gpu, sizeof(char) * cholSize);
    cusolverSpScsrcholFactor(cusolverHandle, ASize, nnzNumber, descrA, AVal, ARow, ACol, d_info, buffer_gpu);

    cudaFree(newIdx);
    cudaFree(newVal);
    cudaFree(ARowTmp);
    cudaFree(ARow);
    cudaFree(ACol);
    cudaFree(AVal);

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
    computeSn << < vertBlocks, threadsPerBlock >> > (sn, dt, dt2_m_1, solverData.X, solverData.V, solverData.dev_ExtForce, masses, m_1_dt2, solverData.numVerts);
    checkCUDAError("computeSn");
    for (int i = 0; i < mcrpSimContext->GetNumIterations(); i++)
    {
        cudaMemset(b, 0, sizeof(float) * solverData.numVerts * 3);
        computeLocal << < tetBlocks, threadsPerBlock >> > (V0, attrib.stiffness_0, b, solverData.inv_Dm, sn, solverData.Tet, solverData.numTets);
        addM_h2Sn << < vertBlocks, threadsPerBlock >> > (b, masses, solverData.numVerts);

        if (mcrpSimContext->IsEigenGlobalSolver())
        {
            cudaMemcpy(bHost, b, sizeof(float) * (solverData.numVerts * 3), cudaMemcpyDeviceToHost);
            Eigen::VectorXf bh = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(bHost, solverData.numVerts * 3);
            Eigen::VectorXf res = cholesky_decomposition_.solve(bh);
            cudaMemcpy(sn, res.data(), sizeof(float) * (solverData.numVerts * 3), cudaMemcpyHostToDevice);
        }
        else
        {
            cusolverSpScsrcholSolve(cusolverHandle, solverData.numVerts * 3, b, sn, d_info, buffer_gpu);
        }
    }

    updateVelPos << < vertBlocks, threadsPerBlock >> > (sn, dtInv, solverData.XTilt, solverData.V, solverData.numVerts);
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


void PdSolver::Laplacian_Smoothing(float blendAlpha)
{
    //cudaMemset(V_sum, 0, sizeof(glm::vec3) * solverData.numVerts);
    //cudaMemset(V_num, 0, sizeof(int) * solverData.numVerts);
    //int blocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    //LaplacianGatherKern << < blocks, threadsPerBlock >> > (V, V_sum, V_num, solverData.numTets, solverData.Tet);
    //LaplacianKern << < (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (V, V_sum, V_num, solverData.numVerts, solverData.Tet, blendAlpha);
}