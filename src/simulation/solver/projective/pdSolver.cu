#include <simulation/solver/linear/cholesky.h>
#include <simulation/solver/linear/jacobi.h>
#include <simulation/solver/projective/pdSolver.h>
#include <simulation/solver/solverUtil.cuh>
#include <simulation/solver/projective/pdUtil.cuh>
#include <fixedBodyData.h>
#include <collision/bvh.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

PdSolver::PdSolver(int threadsPerBlock, const SolverData<float>& solverData) : FEMSolver(threadsPerBlock, solverData)
{
    cudaMalloc((void**)&solverData.ExtForce, sizeof(glm::vec3) * solverData.numVerts);
    cudaMemset(solverData.ExtForce, 0, sizeof(glm::vec3) * solverData.numVerts);
    performanceData = { {"local step", 0.0f}, {"global step", 0.0f}, {"collision handling(fixed)", 0.0f}, {"collision handling(mesh)", 0.0f} };
}

PdSolver::~PdSolver() {
    if (ls) {
        free(ls);
    }
    cudaFree(sn);
    cudaFree(sn_prime);
    cudaFree(b);
    cudaFree(masses);
    free(bHost);

    cudaFree(ARowIdx);
    cudaFree(AColIdx);
    cudaFree(AVal);
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
    cudaMalloc((void**)&sn_prime, sizeof(float) * ASize);
    cudaMalloc((void**)&b, sizeof(float) * ASize);
    cudaMalloc((void**)&masses, sizeof(float) * ASize);

    if (AColIdx && ARowIdx && AVal)
    {
        cudaFree(AColIdx);
        cudaFree(ARowIdx);
        cudaFree(AVal);
    }
    cudaMalloc((void**)&AColIdx, sizeof(int) * len);
    cudaMemset(AColIdx, 0, sizeof(int) * len);

    cudaMalloc((void**)&ARowIdx, sizeof(int) * len);
    cudaMemset(ARowIdx, 0, sizeof(int) * len);

    cudaMalloc((void**)&AVal, sizeof(int) * len);
    cudaMemset(AVal, 0, sizeof(int) * len);

    PdUtil::computeSiTSi << < tetBlocks, threadsPerBlock >> > (ARowIdx, AColIdx, AVal, solverData.V0, solverData.DmInv, solverData.Tet, solverData.mu, solverData.numTets, solverData.numVerts);
    PdUtil::setMDt_2 << < vertBlocks, threadsPerBlock >> > (ARowIdx, AColIdx, AVal, 48 * solverData.numTets, m_1_dt2, solverData.numVerts);

    bHost = (float*)malloc(sizeof(float) * ASize);
    std::vector<int>ARowIdxHost(len);
    std::vector<int>AColIdxHost(len);
    std::vector<float>tmpValHost(len);

    cudaMemcpy(ARowIdxHost.data(), ARowIdx, sizeof(int) * len, cudaMemcpyDeviceToHost);
    cudaMemcpy(AColIdxHost.data(), AColIdx, sizeof(int) * len, cudaMemcpyDeviceToHost);
    cudaMemcpy(tmpValHost.data(), AVal, sizeof(float) * len, cudaMemcpyDeviceToHost);

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
        // transfer A to coo format ARowIdx, AColIdx, AVal
        nnz = A.nonZeros();
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
        cudaMemcpy(AVal, tmpValHost.data(), sizeof(float) * nnz, cudaMemcpyHostToDevice);

        ls = new CholeskySpLinearSolver<float>(threadsPerBlock, ARowIdx, AColIdx, AVal, ASize, nnz);
        jacobiSolver = new JacobiSolver<float>(ASize);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << ", " << "Cholesky decomposition(Eigen) failed" << std::endl;
    }
}

float computeError(thrust::device_ptr<float> sn, thrust::device_ptr<float> sn_prime, int size)
{
    return thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(size),
        [=]__host__ __device__(indexType vertIdx) {
        return (sn_prime[vertIdx] - sn[vertIdx]) * (sn_prime[vertIdx] - sn[vertIdx]);
    },
        0.0,
        thrust::plus<float>()) / size;
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
    cudaMemset(sn_prime, 0, sizeof(float) * solverData.numVerts * 3);
    thrust::device_ptr<float> x_prime_ptr(sn_prime);
    thrust::device_ptr<float> x_ptr(sn);
    glm::vec3 gravity{ 0.0f, -solverParams.gravity * solverParams.softBodyAttr.mass, 0.0f };
    thrust::device_ptr<glm::vec3> dev_ptr(solverData.ExtForce);
    thrust::fill(dev_ptr, dev_ptr + solverData.numVerts, gravity);
    PdUtil::computeSn << < vertBlocks, threadsPerBlock >> > (sn, dt, dt2_m_1, solverData.X, solverData.V, solverData.ExtForce, masses, m_1_dt2, solverData.numVerts);
    float err{ 1 };
    for (int i = 0; i < solverParams.numIterations && sqrt(err) >= solverParams.tol; i++)
    {
        Eigen::VectorXf bh, res;
        performanceData[0].second +=
            measureExecutionTime([&]() {
            cudaMemset(b, 0, sizeof(float) * solverData.numVerts * 3);
            PdUtil::computeLocal << < tetBlocks, threadsPerBlock >> > (solverData.V0, solverData.mu, b, solverData.DmInv, sn, solverData.Tet, solverData.numTets);
            PdUtil::addM_h2Sn << < vertBlocks, threadsPerBlock >> > (b, masses, solverData.numVerts);
                }, perf);
        performanceData[1].second +=
            measureExecutionTime([&]()
                {
                    switch (solverType)
                    {
                    case PdSolver::SolverType::EigenCholesky:
                    {
                        cudaMemcpy(bHost, b, sizeof(float) * (solverData.numVerts * 3), cudaMemcpyDeviceToHost);
                        bh = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(bHost, solverData.numVerts * 3);
                        res = cholesky_decomposition_.solve(bh);
                        cudaMemcpy(sn, res.data(), sizeof(float) * (solverData.numVerts * 3), cudaMemcpyHostToDevice);
                        break;
                    }
                    case PdSolver::SolverType::CuSolverCholesky:
                        ls->Solve(solverData.numVerts * 3, b, sn);
                        break;
                    case PdSolver::SolverType::Jacobi:
                        cudaMemcpy(bHost, b, sizeof(float) * (solverData.numVerts * 3), cudaMemcpyDeviceToHost);
                        bh = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(bHost, solverData.numVerts * 3);
                        res = cholesky_decomposition_.solve(bh);

                        jacobiSolver->Solve(solverData.numVerts * 3, b, sn, AVal, nnz, ARowIdx, AColIdx, res.data());
                        break;
                    default:
                        break;
                    }
                    err = computeError(x_ptr, x_prime_ptr, solverData.numVerts * 3);
                    cudaMemcpy(sn_prime, sn, sizeof(float) * (solverData.numVerts * 3), cudaMemcpyDeviceToDevice);
                }, perf);
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
        performanceData[3].second +=
            measureExecutionTime([&]() {
            solverData.pCollisionDetection->DetectCollision(solverData.numVerts, solverData.numTris, solverData.Tri, solverData.X, solverData.XTilde, solverData.dev_TriFathers, solverData.dev_tIs, solverData.dev_Normals, true);
            int blocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
            CCDKernel << <blocks, threadsPerBlock >> > (solverData.X, solverData.XTilde, solverData.V, solverData.dev_tIs, solverData.dev_Normals, solverParams.muT, solverParams.muN, solverData.numVerts, solverParams.dt);
                }, perf);
    }
    else
        cudaMemcpy(solverData.X, solverData.XTilde, sizeof(glm::vec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
    performanceData[2].second +=
        measureExecutionTime([&]() {
        solverData.pFixedBodies->HandleCollisions(solverData.XTilde, solverData.V, solverData.numVerts, solverParams.muT, solverParams.muN);
            }, perf);
}

void PdSolver::Reset()
{
    Solver::Reset();
    for (auto& pd : performanceData)
    {
        pd.second = 0.0f;
    }
}