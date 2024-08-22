#include <simulation/solver/linear/cholesky.h>
#include <simulation/solver/projective/pdSolver.h>
#include <simulation/solver/projective/pdUtil.cuh>
#include <fixedBodyData.h>
#include <collision/bvh.h>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
    if (ls) {
        free(ls);
    }
    cudaFree(sn);
    cudaFree(b);
    cudaFree(masses);
    free(bHost);
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

    ls = new CholeskySpLinearSolver(threadsPerBlock, AIdx, tmpVal, ASize, len);

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
            ls->Solve(solverData.numVerts * 3, b, sn);
        }
    }

    PdUtil::updateVelPos << < vertBlocks, threadsPerBlock >> > (sn, dtInv, solverData.XTilde, solverData.V, solverData.numVerts);
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
    if (solverParams.handleCollision) {
        solverParams.pCollisionDetection->DetectCollision(solverData.dev_tIs, solverData.dev_Normals);
        int blocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
        PdUtil::CCDKernel << <blocks, threadsPerBlock >> > (solverData.X, solverData.XTilde, solverData.V, solverData.dev_tIs, solverData.dev_Normals, solverParams.muT, solverParams.muN, solverData.numVerts);
    }else
        cudaMemcpy(solverData.X, solverData.XTilde, sizeof(glm::vec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
    solverData.pFixedBodies->HandleCollisions(solverData.XTilde, solverData.V, solverData.numVerts, solverParams.muT, solverParams.muN);
}