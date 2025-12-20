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
#include <utilities.cuh>

struct gravity_force {
    const float g;
    gravity_force(float _g) : g(_g) {}
    __device__ glm::vec3 operator()(float mass) const { return  glm::vec3{ 0.0f, -g * mass, 0.0f }; }
};

float computeError(thrust::device_ptr<float> sn, thrust::device_ptr<float> sn_old, int size);

PdSolver::PdSolver(int threadsPerBlock, const SolverData<float>& solverData) : FEMSolver(threadsPerBlock, solverData), solverType(SolverType::Jacobi)
{
    cudaMalloc((void**)&solverData.ExtForce, sizeof(glm::vec3) * solverData.numVerts);
    cudaMemset(solverData.ExtForce, 0, sizeof(glm::vec3) * solverData.numVerts);
    performanceData = { {"local step", 0.0f}, {"global step", 0.0f}, {"collision handling(fixed)", 0.0f}, {"collision handling(mesh)", 0.0f} };
}

PdSolver::~PdSolver() {
if (sn) cudaFree(sn);
    if (sn_old) cudaFree(sn_old);
    if (b) cudaFree(b);
    if (massDt_2s) cudaFree(massDt_2s);
    if (bHost) free(bHost);
    if (next_x) cudaFree(next_x);
    if (prev_x) cudaFree(prev_x);
    if (matrix_diag) cudaFree(matrix_diag);
}

void PdSolver::SolverPrepare(SolverData<float>& solverData, const SolverParams<float>& solverParams)
{
    if (sn) cudaFree(sn);
    if (sn_old) cudaFree(sn_old);
    if (next_x) cudaFree(next_x);
    if (prev_x) cudaFree(prev_x);
    if (b) cudaFree(b);
    if (massDt_2s) cudaFree(massDt_2s);
    if (matrix_diag) cudaFree(matrix_diag);
    if (bHost) free(bHost);
    int vertBlocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    int tetBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    float dt = solverParams.dt;
    int len = solverData.numVerts * 3 + 48 * solverData.numTets;
    int ASize = 3 * solverData.numVerts;
    cudaMalloc((void**)&sn, sizeof(float) * ASize);
    cudaMalloc((void**)&sn_old, sizeof(float) * ASize);
    cudaMalloc((void**)&next_x, sizeof(float) * ASize);
    cudaMalloc((void**)&prev_x, sizeof(float) * ASize);
    cudaMalloc((void**)&b, sizeof(float) * ASize);
    cudaMalloc((void**)&massDt_2s, sizeof(float) * solverData.numVerts);
    cudaMalloc((void**)&matrix_diag, sizeof(float) * solverData.numVerts);
    cudaMemset(matrix_diag, 0, sizeof(float) * solverData.numVerts);

    int* AColIdx, * ARowIdx;
    cudaMalloc((void**)&AColIdx, sizeof(int) * len);
    cudaMemset(AColIdx, 0, sizeof(int) * len);
    cudaMalloc((void**)&ARowIdx, sizeof(int) * len);
    cudaMemset(ARowIdx, 0, sizeof(int) * len);

    float* AVal;
    cudaMalloc((void**)&AVal, sizeof(int) * len);
    cudaMemset(AVal, 0, sizeof(int) * len);

    size_t offset = 0;
    PdUtil::computeSiTSi << < tetBlocks, threadsPerBlock >> > (ARowIdx, AColIdx, AVal, matrix_diag, solverData.V0, solverData.DmInv, solverData.Tet, solverData.mu, solverData.numTets, solverData.numVerts);
    offset += 48 * solverData.numTets;
    PdUtil::setMDt_2 << < vertBlocks, threadsPerBlock >> > (solverData.numVerts, ARowIdx, AColIdx, AVal, offset, solverData.mass, dt * dt, massDt_2s, solverData.DBC, positional_weight);

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
        cudaMemcpy(AVal, tmpValHost.data(), sizeof(float) * nnz, cudaMemcpyHostToDevice);

        ls = std::make_unique<CholeskySpLinearSolver<float>>(threadsPerBlock, ARowIdx, AColIdx, AVal, ASize, nnz);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << ", " << "Cholesky decomposition(Eigen) failed" << std::endl;
    }
    cudaMemcpy(solverData.DBCX, solverData.X0, sizeof(glm::vec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);

    cudaFree(ARowIdx);
    cudaFree(AColIdx);
    cudaFree(AVal);
}

bool PdSolver::SolverStep(SolverData<float>& solverData, const SolverParams<float>& solverParams)
{
    float dt = solverParams.dt;
    float const dtInv = 1.0f / dt;
    float const dt2 = dt * dt;
    float const dt2Inv = dtInv * dtInv;

    int vertBlocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    int vert3Blocks = (solverData.numVerts * 3 + threadsPerBlock - 1) / threadsPerBlock;
    int tetBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    //inspectGLM(solverData.DBCX, solverData.numVerts, "DBCX");
    thrust::device_ptr<float> x_prime_ptr(prev_x);
    thrust::device_ptr<float> x_ptr(sn);
    thrust::transform(thrust::device_pointer_cast(solverData.mass), thrust::device_pointer_cast(solverData.mass) + solverData.numVerts,
        thrust::device_pointer_cast(solverData.ExtForce), gravity_force(solverParams.gravity));
    PdUtil::setMDt_2MoreDBC << < vertBlocks, threadsPerBlock >> > (solverData.numVerts, solverData.mass, dt * dt, massDt_2s, solverData.moreDBC, solverData.DBC);
    PdUtil::computeSn << < vertBlocks, threadsPerBlock >> > (solverData.numVerts, sn, dt, massDt_2s, solverData.X, solverData.V, solverData.ExtForce, solverData.moreDBC, solverData.OffsetX, solverData.DBCX, solverData.mouseSelection.target);
    cudaMemcpy(sn_old, sn, sizeof(float) * (solverData.numVerts * 3), cudaMemcpyDeviceToDevice);
    if (solverType == PdSolver::SolverType::Jacobi)
        cudaMemcpy(prev_x, sn, sizeof(float) * (solverData.numVerts * 3), cudaMemcpyDeviceToDevice);
    else
        cudaMemset(prev_x, 0, solverData.numVerts * 3);
    float err{ 1 };
    for (int i = 0; i < solverParams.numIterations && sqrt(err) >= solverParams.tol; i++)
    {
        performanceData[0].second +=
            measureExecutionTime([&]() {
            PdUtil::addM_h2Sn << < vertBlocks, threadsPerBlock >> > (b, sn_old, massDt_2s, solverData.numVerts);
            PdUtil::computeLocal << < tetBlocks, threadsPerBlock >> > (solverData.V0, solverData.mu, b, solverData.DmInv, sn, solverData.Tet, solverData.numTets, solverType == PdSolver::SolverType::Jacobi);
            if (solverData.numDBC > 0)
                PdUtil::computeDBCLocal << < vertBlocks, threadsPerBlock >> > (solverData.numVerts, solverData.DBC, solverData.moreDBC, solverData.DBCX, positional_weight * dt2Inv, b);
                }, perf);
        performanceData[1].second +=
            measureExecutionTime([&]()
                {
                    switch (solverType)
                    {
                    case PdSolver::SolverType::EigenCholesky:
                    {
                        cudaMemcpy(bHost, b, sizeof(float) * (solverData.numVerts * 3), cudaMemcpyDeviceToHost);
                        Eigen::VectorXf bh = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(bHost, solverData.numVerts * 3);
                        Eigen::VectorXf res = cholesky_decomposition_.solve(bh);
                        cudaMemcpy(sn, res.data(), sizeof(float) * (solverData.numVerts * 3), cudaMemcpyHostToDevice);
                        break;
                    }
                    case PdSolver::SolverType::CuSolverCholesky:
                    {
                        ls->Solve(solverData.numVerts * 3, b, sn);
                        err = computeError(x_ptr, x_prime_ptr, solverData.numVerts * 3);
                        cudaMemcpy(prev_x, sn, sizeof(float) * (solverData.numVerts * 3), cudaMemcpyDeviceToDevice);
                        break;
                    }
                    case PdSolver::SolverType::Jacobi:
                        // correct massDt_2 with moreDBC
                        PdUtil::getErrorKern << < vertBlocks, threadsPerBlock >> > (solverData.numVerts, next_x, b, massDt_2s, sn, matrix_diag, solverData.moreDBC);
                        if (i <= 10)		omega = 1;
                        else if (i == 11)	omega = 2 / (2 - solverParams.rho * solverParams.rho);
                        else			omega = 4 / (4 - solverParams.rho * solverParams.rho * omega);
                        PdUtil::chebyshevKern << < vert3Blocks, threadsPerBlock >> > (solverData.numVerts * 3, next_x, prev_x, sn, omega);
                        break;
                    default:
                        break;
                    }
                }, perf);
    }
    PdUtil::updateVelPos << < vertBlocks, threadsPerBlock >> > (sn, dtInv, solverData.XTilde, solverData.V, solverData.numVerts, solverData.moreDBC);
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

float computeError(thrust::device_ptr<float> sn, thrust::device_ptr<float> sn_old, int size)
{
    return thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(size),
        [=]__host__ __device__(indexType vertIdx) {
        return (sn_old[vertIdx] - sn[vertIdx]) * (sn_old[vertIdx] - sn[vertIdx]);
    },
        0.0,
        thrust::plus<float>()) / size;
}
