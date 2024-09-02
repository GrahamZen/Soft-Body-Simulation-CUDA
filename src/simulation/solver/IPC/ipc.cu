#include <IPC/ipc.h>
#include <collision/bvh.h>
#include <linear/choleskyImmed.h>
#include <fixedBodyData.h>
#include <solver/solverUtil.cuh>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>

IPCSolver::IPCSolver(int threadsPerBlock, const SolverData<double>& solverData, double tol)
    : numVerts(solverData.numVerts), tolerance(tol), FEMSolver(threadsPerBlock, solverData),
    energy(solverData), linearSolver(new CholeskySpImmedSolver<double>(solverData.numVerts * 3))
{
    cudaMalloc(&p, sizeof(double) * solverData.numVerts * 3);
    cudaMalloc(&xTmp, sizeof(glm::dvec3) * solverData.numVerts);
    cudaMalloc(&x_n, sizeof(glm::dvec3) * solverData.numVerts);
}

IPCSolver::~IPCSolver()
{
    cudaFree(p);
    cudaFree(xTmp);
    cudaFree(x_n);
}

void IPCSolver::Update(SolverData<double>& solverData, SolverParams& solverParams)
{
    SolverStep(solverData, solverParams);
    if (solverParams.handleCollision) {
        cudaMemcpy(solverData.XTilde, solverData.X, sizeof(glm::dvec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
        cudaMemcpy(solverData.X, x_n, sizeof(glm::dvec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
        solverParams.pCollisionDetection->DetectCollision(solverData.dev_tIs, solverData.dev_Normals);
        int blocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
        IPCCDKernel << <blocks, threadsPerBlock >> > (solverData.X, solverData.XTilde, solverData.V, solverData.dev_tIs, solverData.dev_Normals, solverParams.muT, solverParams.muN, solverData.numVerts);
    }
}

void IPCSolver::SolverPrepare(SolverData<double>& solverData, SolverParams& solverParams)
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

    __global__ void DOFEliminationHessKernel(int* hessianRowIdx, int* hessianColIdx, double* hessianVal, int nnz, indexType* DBC, int numDBC)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= nnz) return;

        int row = hessianRowIdx[idx];
        int col = hessianColIdx[idx];
        for (int i = 0; i < numDBC; i++)
        {
            if (DBC[i] == row / 3 || DBC[i] == col / 3) {
                hessianVal[idx] = (row == col);
                if (row != col)
                    break;
            }
        }
    }
    __global__ void DOFEliminationGradKernel(double* gradient, int numVerts, indexType* DBC, int numDBC)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= numVerts) return;

        for (int i = 0; i < numDBC; i++)
        {
            if (DBC[i] == idx)
            {
                gradient[idx * 3] = 0;
                gradient[idx * 3 + 1] = 0;
                gradient[idx * 3 + 2] = 0;
                break;
            }
        }
    }
}

void IPCSolver::SolverStep(SolverData<double>& solverData, SolverParams& solverParams)
{
    double h = solverParams.dt;
    double h2 = h * h;
    int blocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    cudaMemcpy(x_n, solverData.X, sizeof(glm::dvec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
    IPC::computeXTilde << <blocks, threadsPerBlock >> > (solverData.XTilde, solverData.X, solverData.V, h, solverData.numVerts);
    double E_last = 0;
    E_last = energy.Val(solverData.X, solverData, h2);

    SearchDirection(solverData, h2);
    while (!EndCondition(h)) {
        double alpha = energy.InitStepSize(solverData, p);
        while (true) {
            IPC::computeXMinusAP << <blocks, threadsPerBlock >> > (xTmp, solverData.X, p, alpha, solverData.numVerts);
            double E = energy.Val(xTmp, solverData, h2);
            if (E > E_last) {
                alpha /= 2;
            }
            else {
                break;
            }
        }
        cudaMemcpy(solverData.X, xTmp, sizeof(glm::dvec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
        E_last = energy.Val(solverData.X, solverData, h2);
        SearchDirection(solverData, h2);
    }
    IPC::updateVel << <blocks, threadsPerBlock >> > (solverData.X, x_n, solverData.V, 1.0 / h, solverData.numVerts);
}

void IPCSolver::SearchDirection(SolverData<double>& solverData, double h2)
{
    energy.Gradient(solverData, h2);
    energy.Hessian(solverData, h2);
    DOFElimination(solverData);
    linearSolver->Solve(solverData.numVerts * 3, energy.gradient, p, energy.hessianVal, energy.nnz, energy.hessianRowIdx, energy.hessianColIdx);
}

void IPCSolver::DOFElimination(SolverData<double>& solverData)
{
    int blocks = (energy.nnz + threadsPerBlock - 1) / threadsPerBlock;
    IPC::DOFEliminationHessKernel << <blocks, threadsPerBlock >> > (energy.hessianRowIdx, energy.hessianColIdx, energy.hessianVal, energy.nnz, solverData.DBC, solverData.numDBC);
    blocks = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    IPC::DOFEliminationGradKernel << <blocks, threadsPerBlock >> > (energy.gradient, numVerts, solverData.DBC, solverData.numDBC);
}

bool IPCSolver::EndCondition(double h)
{
    thrust::device_ptr<double> dev_ptr(p);
    double inf_norm = thrust::transform_reduce(dev_ptr, dev_ptr + numVerts * 3,
        [] __host__ __device__(double x) { return abs(x); }, 0.0, thrust::maximum<double>()) / h;

    return inf_norm < tolerance;
}
