#include <energy/ip.h>
#include <energy/corotated.h>
#include <cuda_runtime.h>

IPEnergy::IPEnergy(const SolverData<double>& solverData, double dHat) : inertia(solverData, nnz, solverData.numVerts, solverData.mass),
elastic(new CorotatedEnergy<double>(solverData, nnz)), barrier(solverData, nnz, dHat)
{
    cudaMalloc(&gradient, sizeof(double) * solverData.numVerts * 3);
    cudaMalloc(&hessianVal, sizeof(double) * nnz);
    cudaMalloc(&hessianRowIdx, sizeof(int) * nnz);
    cudaMalloc(&hessianColIdx, sizeof(int) * nnz);
    inertia.SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
    barrier.SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
    elastic->SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
}

IPEnergy::~IPEnergy()
{
    cudaFree(gradient);
    cudaFree(hessianVal);
    cudaFree(hessianRowIdx);
    cudaFree(hessianColIdx);
}

double IPEnergy::Val(const glm::dvec3* Xs, const SolverData<double>& solverData, double h2) const
{
    return inertia.Val(Xs, solverData) + h2 * (gravity.Val(Xs, solverData) + elastic->Val(Xs, solverData) + barrier.Val(Xs, solverData));
}

void IPEnergy::Gradient(const SolverData<double>& solverData, double h2) const
{
    cudaMemset(gradient, 0, sizeof(double) * solverData.numVerts * 3);
    inertia.Gradient(gradient, solverData, 1);
    gravity.Gradient(gradient, solverData, h2);
    elastic->Gradient(gradient, solverData, h2);
    barrier.Gradient(gradient, solverData, h2);
}

void IPEnergy::Hessian(const SolverData<double>& solverData, double h2) const
{
    cudaMemset(hessianVal, 0, sizeof(double) * nnz);
    inertia.Hessian(solverData, 1);
    gravity.Hessian(solverData, h2);
    elastic->Hessian(solverData, h2);
    barrier.Hessian(solverData, h2);
}

double IPEnergy::InitStepSize(const SolverData<double>& solverData, double* p) const
{
    return barrier.InitStepSize(solverData, p);
}
