#include <energy/ip.h>
#include <energy/corotated.h>
#include <collision/bvh.h>

IPEnergy::IPEnergy(const SolverData<double>& solverData) :inertia(solverData, nnz, solverData.numVerts, solverData.mass),
elastic(new CorotatedEnergy<double>(solverData, nnz)), implicitBarrier(solverData, nnz), barrier(solverData, nnz)
{
    cudaMalloc((void**)&gradient, sizeof(double) * solverData.numVerts * 3);
    cudaMalloc((void**)&hessianVal, sizeof(double) * nnz);
    cudaMalloc((void**)&hessianRowIdx, sizeof(int) * nnz);
    cudaMalloc((void**)&hessianColIdx, sizeof(int) * nnz);
    inertia.SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
    implicitBarrier.SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
    elastic->SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
    barrier.SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
}

IPEnergy::~IPEnergy()
{
    cudaFree(gradient);
    cudaFree(hessianVal);
    cudaFree(hessianRowIdx);
    cudaFree(hessianColIdx);
}

double IPEnergy::Val(const glm::dvec3* Xs, const SolverData<double>& solverData, const SolverParams<double>& solverParams, double h2) const
{
    return inertia.Val(Xs, solverData, solverParams) + h2 * (gravity.Val(Xs, solverData, solverParams) + elastic->Val(Xs, solverData, solverParams) + implicitBarrier.Val(Xs, solverData, solverParams) + barrier.Val(Xs, solverData, solverParams));
}

void IPEnergy::Gradient(const SolverData<double>& solverData, const SolverParams<double>& solverParams, double h2) const
{
    cudaMemset(gradient, 0, sizeof(double) * solverData.numVerts * 3);
    inertia.Gradient(gradient, solverData, solverParams, 1);
    gravity.Gradient(gradient, solverData, solverParams, h2);
    elastic->Gradient(gradient, solverData, solverParams, h2);
    implicitBarrier.Gradient(gradient, solverData, solverParams, h2);
    barrier.Gradient(gradient, solverData, solverParams, h2);
}

void IPEnergy::Hessian(const SolverData<double>& solverData, const SolverParams<double>& solverParams, double h2) const
{
    cudaMemset(hessianVal, 0, sizeof(double) * NNZ(solverData));
    inertia.Hessian(solverData, solverParams, 1);
    gravity.Hessian(solverData, solverParams, h2);
    elastic->Hessian(solverData, solverParams, h2);
    implicitBarrier.Hessian(solverData, solverParams, h2);
    barrier.Hessian(solverData, solverParams, h2);
}

double IPEnergy::InitStepSize(SolverData<double>& solverData, const SolverParams<double>& solverParams, double* p, glm::tvec3<double>* XTmp) const
{
    return std::min(1.0, std::min(implicitBarrier.InitStepSize(solverData, p), barrier.InitStepSize(solverData, p, XTmp)));
}

int IPEnergy::NNZ(const SolverData<double>& solverData) const
{
    return inertia.NNZ(solverData) + implicitBarrier.NNZ(solverData) + elastic->NNZ(solverData) + barrier.NNZ(solverData);
}