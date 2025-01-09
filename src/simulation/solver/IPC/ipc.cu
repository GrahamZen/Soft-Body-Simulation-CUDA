#include <IPC/ipc.h>
#include <collision/bvh.h>
#include <linear/choleskyImmed.h>
#include <utilities.cuh>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <fstream>
#include <solverUtil.cuh>


IPCSolver::IPCSolver(int threadsPerBlock, const SolverData<double>& solverData)
    : numVerts(solverData.numVerts), FEMSolver(threadsPerBlock, solverData),
    energy(solverData), linearSolver(new CholeskySpImmedSolver<double>(solverData.numVerts * 3))
{
    cudaMalloc((void**)&p, sizeof(double) * solverData.numVerts * 3);
    cudaMalloc((void**)&xTmp, sizeof(glm::dvec3) * solverData.numVerts);
    cudaMalloc((void**)&x_n, sizeof(glm::dvec3) * solverData.numVerts);
    performanceData = { {"Init search dir", 0.0f},{"Line search", 0.0f} ,{"CCD", 0.0f} ,{"Compute search dir", 0.0f} };
}

IPCSolver::~IPCSolver()
{
    cudaFree(p);
    cudaFree(xTmp);
    cudaFree(x_n);
}

void IPCSolver::Update(SolverData<double>& solverData, const SolverParams<double>& solverParams)
{
    if (failed) return;
    if (!SolverStep(solverData, solverParams)) {
        std::cout << "IPC Solver did not converge" << std::endl;
        failed = true;
    }
}

void IPCSolver::Reset()
{
    Solver::Reset();
    failed = false;
}

void IPCSolver::SolverPrepare(SolverData<double>& solverData, const SolverParams<double>& solverParams)
{
}
namespace IPC {

    __global__ void computeXTilde(glm::dvec3* xTilde, const glm::dvec3* x, glm::dvec3* v, double dt, int numVerts)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= numVerts) return;

        xTilde[idx] = x[idx] + dt * v[idx];
    }
    __global__ void computeXMinusAP(glm::dvec3* xPlusAP, const glm::dvec3* x, const double* p, double alpha, int numVerts)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= numVerts) return;

        xPlusAP[idx].x = x[idx].x - alpha * p[idx * 3];
        xPlusAP[idx].y = x[idx].y - alpha * p[idx * 3 + 1];
        xPlusAP[idx].z = x[idx].z - alpha * p[idx * 3 + 2];
    }

    __global__ void updateVel(const glm::dvec3* x, const glm::dvec3* x_n, glm::dvec3* v, double dtInv, int numVerts)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numVerts)
        {
            v[index] = (x[index] - x_n[index]) * dtInv;
        }
    }

    __global__ void DOFEliminationHessKernel(int* hessianRowIdx, int* hessianColIdx, double* hessianVal, int nnz, indexType* DBCIdx, int numDBC)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= nnz) return;

        int row = hessianRowIdx[idx];
        int col = hessianColIdx[idx];
        for (int i = 0; i < numDBC; i++)
        {
            if (DBCIdx[i] == row / 3 || DBCIdx[i] == col / 3) {
                hessianVal[idx] = (row == col);
                if (row != col)
                    break;
            }
        }
    }
    __global__ void DOFEliminationGradKernel(double* gradient, int numVerts, indexType* DBCIdx, int numDBC)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= numVerts) return;

        for (int i = 0; i < numDBC; i++)
        {
            if (DBCIdx[i] == idx)
            {
                gradient[idx * 3] = 0;
                gradient[idx * 3 + 1] = 0;
                gradient[idx * 3 + 2] = 0;
                break;
            }
        }
    }
}

bool IPCSolver::SolverStep(SolverData<double>& solverData, const SolverParams<double>& solverParams)
{
    double h = solverParams.dt;
    double h2 = h * h;
    int blocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    double E_last = 0;
    performanceData[0].second +=
        measureExecutionTime([&]() {
        cudaMemcpy(x_n, solverData.X, sizeof(glm::dvec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
        IPC::computeXTilde << <blocks, threadsPerBlock >> > (solverData.XTilde, solverData.X, solverData.V, h, solverData.numVerts);
        solverData.pCollisionDetection->UpdateQueries(solverData.numVerts, solverData.numTris, solverData.Tri, solverData.X, solverData.dev_TriFathers, solverParams.dhat);
        E_last = energy.Val(solverData.X, solverData, solverParams, h2);

        SearchDirection(solverData, solverParams, h2);
        solverData.pCollisionDetection->UpdateDirection(p);
        solverData.pCollisionDetection->UpdateX(solverData.X);
            }, perf);
    int maxIter = solverParams.maxIterations;
    int iter = 0;
    while (!EndCondition(h, solverParams.tol)) {
        if (++iter > maxIter) {
            return false;
        }
        performanceData[1].second +=
            measureExecutionTime([&]() {
            IPC::computeXMinusAP << <blocks, threadsPerBlock >> > (xTmp, solverData.X, p, 1, solverData.numVerts);
            double alpha = energy.InitStepSize(solverData, solverParams, p, xTmp);
            while (true) {
                IPC::computeXMinusAP << <blocks, threadsPerBlock >> > (xTmp, solverData.X, p, alpha, solverData.numVerts);
                solverData.pCollisionDetection->UpdateQueries(solverData.numVerts, solverData.numTris, solverData.Tri, xTmp, solverData.dev_TriFathers, solverParams.dhat);
                double E = energy.Val(xTmp, solverData, solverParams, h2);
                if (E > E_last)
                    alpha /= 2;
                else
                    break;
            }
            cudaMemcpy(solverData.X, xTmp, sizeof(glm::dvec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
                }, perf);
        performanceData[2].second +=
            measureExecutionTime([&]() {
            solverData.pCollisionDetection->UpdateQueries(solverData.numVerts, solverData.numTris, solverData.Tri, solverData.X, solverData.dev_TriFathers, solverParams.dhat);
                }, perf);
        performanceData[3].second +=
            measureExecutionTime([&]() {
            E_last = energy.Val(solverData.X, solverData, solverParams, h2);
            SearchDirection(solverData, solverParams, h2);
                }, perf);
    }
    IPC::updateVel << <blocks, threadsPerBlock >> > (solverData.X, x_n, solverData.V, 1.0 / h, solverData.numVerts);
    return true;
}

void IPCSolver::SearchDirection(SolverData<double>& solverData, const SolverParams<double>& solverParams, double h2)
{
    energy.GradientHessian(solverData, solverParams, h2);
    DOFElimination(solverData);
    linearSolver->Solve(solverData.numVerts * 3, energy.gradient, p, energy.hessianVal, energy.NNZ(solverData), energy.hessianRowIdx, energy.hessianColIdx, (double*)solverData.X);
}

void IPCSolver::DOFElimination(SolverData<double>& solverData)
{
    int blocks = (energy.NNZ(solverData) + threadsPerBlock - 1) / threadsPerBlock;
    IPC::DOFEliminationHessKernel << <blocks, threadsPerBlock >> > (energy.hessianRowIdx, energy.hessianColIdx, energy.hessianVal, energy.NNZ(solverData), solverData.DBCIdx, solverData.numDBC);
    blocks = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    IPC::DOFEliminationGradKernel << <blocks, threadsPerBlock >> > (energy.gradient, numVerts, solverData.DBCIdx, solverData.numDBC);
}

bool IPCSolver::EndCondition(double h, double tolerance)
{
    thrust::device_ptr<double> dev_ptr(p);
    double inf_norm = thrust::transform_reduce(dev_ptr, dev_ptr + numVerts * 3,
        [] __host__ __device__(double x) { return abs(x); }, 0.0, thrust::maximum<double>()) / h;

    return inf_norm < tolerance;
}
