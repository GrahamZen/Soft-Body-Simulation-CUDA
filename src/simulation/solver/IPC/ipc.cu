#include <IPC/ipc.h>
#include <cuda_runtime.h>

IPCSolver::IPCSolver(int threadsPerBlock, const SolverData<double>& solverData)
    :FEMSolver(threadsPerBlock, solverData),
    inertia(solverData, nnz, solverData.numVerts, solverData.mass),
    elastic(new CorotatedEnergy<double>(solverData, nnz))
{
    cudaMalloc(&gradient, sizeof(double) * solverData.numVerts * 3);
    cudaMalloc(&hessianVal, sizeof(double) * nnz);
    cudaMalloc(&hessianRowIdx, sizeof(int) * nnz);
    cudaMalloc(&hessianColIdx, sizeof(int) * nnz);

    inertia.SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
    elastic->SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
}

IPCSolver::~IPCSolver()
{
    cudaFree(gradient);
    cudaFree(hessianVal);
    cudaFree(hessianRowIdx);
    cudaFree(hessianColIdx);
}

void IPCSolver::Update(SolverData<double>& solverData, SolverParams& solverParams)
{
    SolverStep(solverData, solverParams);
}

void IPCSolver::SolverPrepare(SolverData<double>& solverData, SolverParams& solverParams)
{
}

void IPCSolver::SolverStep(SolverData<double>& solverData, SolverParams& solverParams)
{
    double h2 = solverParams.dt * solverParams.dt;
    inertia.Gradient(gradient, solverData, 1);
    gravity.Gradient(gradient, solverData, h2);
    elastic->Gradient(gradient, solverData, h2);
    inertia.Hessian(solverData, 1);
    gravity.Hessian(solverData, h2);
    elastic->Hessian(solverData, h2);
}
