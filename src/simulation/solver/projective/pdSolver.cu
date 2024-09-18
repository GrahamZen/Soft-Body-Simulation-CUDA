#include <simulation/solver/linear/cholesky.h>
#include <simulation/solver/projective/pdSolver.h>
#include <simulation/solver/solverUtil.cuh>
#include <simulation/solver/projective/pdUtil.cuh>
#include <fixedBodyData.h>
#include <collision/bvh.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

PdSolver::PdSolver(int threadsPerBlock, const SolverData<float>& solverData) : FEMSolver(threadsPerBlock, solverData)
{
    cudaMalloc((void**)&solverData.ExtForce, sizeof(glm::vec3) * solverData.numVerts);
    cudaMemset(solverData.ExtForce, 0, sizeof(glm::vec3) * solverData.numVerts);
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

void PdSolver::SolverPrepare(SolverData<float>& solverData, const SolverParams<float>& solverParams)
{
    int vertBlocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    int tetBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    float dt = solverParams.dt;
    float const m_1_dt2 = solverParams.softBodyAttr.mass / (dt * dt);
    int len = solverData.numVerts * 3 + 48 * solverData.numTets;
    int ASize = 3 * solverData.numVerts;

    cudaMalloc((void**)&sn, sizeof(float) * ASize);
    cudaMalloc((void**)&b, sizeof(float) * ASize);
    cudaMalloc((void**)&masses, sizeof(float) * ASize);

    int* AColIdx;
    cudaMalloc((void**)&AColIdx, sizeof(int) * len);
    cudaMemset(AColIdx, 0, sizeof(int) * len);

    int* ARowIdx;
    cudaMalloc((void**)&ARowIdx, sizeof(int) * len);
    cudaMemset(ARowIdx, 0, sizeof(int) * len);

    float* tmpVal;
    cudaMalloc((void**)&tmpVal, sizeof(int) * len);
    cudaMemset(tmpVal, 0, sizeof(int) * len);

    PdUtil::computeSiTSi << < tetBlocks, threadsPerBlock >> > (ARowIdx, AColIdx, tmpVal, solverData.V0, solverData.DmInv, solverData.Tet, solverParams.softBodyAttr.mu, solverData.numTets, solverData.numVerts);
    PdUtil::setMDt_2 << < vertBlocks, threadsPerBlock >> > (ARowIdx, AColIdx, tmpVal, 48 * solverData.numTets, m_1_dt2, solverData.numVerts);

    bHost = (float*)malloc(sizeof(float) * ASize);
    // int* AIdxHost = (int*)malloc(sizeof(int) * len);
    std::vector<int>ARowIdxHost(len);
    std::vector<int>AColIdxHost(len);
    std::vector<float>tmpValHost(len);

    // cudaMemcpy(AIdxHost, AIdx, sizeof(int) * len, cudaMemcpyDeviceToHost);
    cudaMemcpy(ARowIdxHost.data(), ARowIdx, sizeof(int) * len, cudaMemcpyDeviceToHost);
    cudaMemcpy(AColIdxHost.data(), AColIdx, sizeof(int) * len, cudaMemcpyDeviceToHost);
    cudaMemcpy(tmpValHost.data(), tmpVal, sizeof(float) * len, cudaMemcpyDeviceToHost);

    try
    {
        std::vector<Eigen::Triplet<float>> A_triplets;
        for (auto i = 0; i < len; ++i)
        {
            A_triplets.push_back({ ARowIdxHost[i], AColIdxHost[i], tmpValHost[i] });
            const auto& triplet = A_triplets.back();
            if (triplet.row() < 0 || triplet.row() >= ASize ||
                triplet.col() < 0 || triplet.col() >= ASize) {
                throw std::invalid_argument("Triplet contains invalid row or column index.");
            }
        }
        Eigen::SparseMatrix<float> A(ASize, ASize);

        A.setFromTriplets(A_triplets.begin(), A_triplets.end());
        cholesky_decomposition_.compute(A);
        A.makeCompressed();
        // transfer A to coo format ARowIdx, AColIdx, tmpVal
        int nnz = A.nonZeros();
        if (nnz != len) {
            ARowIdxHost.resize(nnz);
            AColIdxHost.resize(nnz);
            tmpValHost.resize(nnz);
        }

        int idx = 0;
        for (int k = 0; k < A.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<float>::InnerIterator it(A, k); it; ++it)
            {
                ARowIdxHost[idx] = it.row();
                AColIdxHost[idx] = it.col();
                tmpValHost[idx] = it.value();
                idx++;
            }
        }
        cudaMemcpy(ARowIdx, ARowIdxHost.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(AColIdx, AColIdxHost.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(tmpVal, tmpValHost.data(), sizeof(float) * nnz, cudaMemcpyHostToDevice);

        ls = new CholeskySpLinearSolver<float>(threadsPerBlock, ARowIdx, AColIdx, tmpVal, ASize, nnz);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << ", " << "Cholesky decomposition(Eigen) failed" << std::endl;
    }


    cudaFree(ARowIdx);
    cudaFree(AColIdx);
    cudaFree(tmpVal);
}


bool PdSolver::SolverStep(SolverData<float>& solverData, const SolverParams<float>& solverParams)
{
    float dt = solverParams.dt;
    float const dtInv = 1.0f / dt;
    float const dt2 = dt * dt;
    float const dt2_m_1 = dt2 / solverParams.softBodyAttr.mass;
    float const m_1_dt2 = 1.f / dt2_m_1;

    int vertBlocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    int tetBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;

    glm::vec3 gravity{ 0.0f, -solverParams.gravity * solverParams.softBodyAttr.mass, 0.0f };
    thrust::device_ptr<glm::vec3> dev_ptr(solverData.ExtForce);
    thrust::fill(dev_ptr, dev_ptr + solverData.numVerts, gravity);
    PdUtil::computeSn << < vertBlocks, threadsPerBlock >> > (sn, dt, dt2_m_1, solverData.X, solverData.V, solverData.ExtForce, masses, m_1_dt2, solverData.numVerts);
    for (int i = 0; i < solverParams.numIterations; i++)
    {
        cudaMemset(b, 0, sizeof(float) * solverData.numVerts * 3);
        PdUtil::computeLocal << < tetBlocks, threadsPerBlock >> > (solverData.V0, solverParams.softBodyAttr.mu, b, solverData.DmInv, sn, solverData.Tet, solverData.numTets);
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
    return true;
}

void PdSolver::Update(SolverData<float>& solverData, const SolverParams<float>& solverParams)
{
    if (!solverReady)
    {
        SolverPrepare(solverData, solverParams);
        solverReady = true;
    }
    SolverStep(solverData, solverParams);
    if (solverParams.handleCollision) {
        solverData.pCollisionDetection->DetectCollision(solverData.numVerts, solverData.numTris, solverData.Tri, solverData.X, solverData.XTilde, solverData.dev_TriFathers, solverData.dev_tIs, solverData.dev_Normals, true);
        int blocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
        CCDKernel << <blocks, threadsPerBlock >> > (solverData.X, solverData.XTilde, solverData.V, solverData.dev_tIs, solverData.dev_Normals, solverParams.muT, solverParams.muN, solverData.numVerts, solverParams.dt);
    }
    else
        cudaMemcpy(solverData.X, solverData.XTilde, sizeof(glm::vec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
    solverData.pFixedBodies->HandleCollisions(solverData.XTilde, solverData.V, solverData.numVerts, solverParams.muT, solverParams.muN);
}