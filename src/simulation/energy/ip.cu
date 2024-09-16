#include <energy/ip.h>
#include <energy/corotated.h>
#include <collision/bvh.h>

IPEnergy::IPEnergy(const SolverData<double>& solverData, double dHat) : inertia(solverData, nnz, solverData.numVerts, solverData.mass),
elastic(new CorotatedEnergy<double>(solverData, nnz)), implicitBarrier(solverData, nnz, dHat), barrier(solverData, nnz, dHat)
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

double IPEnergy::Val(const glm::dvec3* Xs, const SolverData<double>& solverData, double h2) const
{
    return inertia.Val(Xs, solverData) + h2 * (gravity.Val(Xs, solverData) + elastic->Val(Xs, solverData) + implicitBarrier.Val(Xs, solverData) + barrier.Val(Xs, solverData));
}

void IPEnergy::Gradient(const SolverData<double>& solverData, double h2) const
{
    cudaMemset(gradient, 0, sizeof(double) * solverData.numVerts * 3);
    inertia.Gradient(gradient, solverData, 1);
    gravity.Gradient(gradient, solverData, h2);
    elastic->Gradient(gradient, solverData, h2);
    implicitBarrier.Gradient(gradient, solverData, h2);
    barrier.Gradient(gradient, solverData, h2);
}

void IPEnergy::Hessian(const SolverData<double>& solverData, double h2) const
{
    cudaMemset(hessianVal, 0, sizeof(double) * NNZ(solverData));
    inertia.Hessian(solverData, 1);
    gravity.Hessian(solverData, h2);
    elastic->Hessian(solverData, h2);
    implicitBarrier.Hessian(solverData, h2);
    barrier.Hessian(solverData, h2);
}

double IPEnergy::InitStepSize(SolverData<double>& solverData, double* p, glm::tvec3<double>* XTmp) const
{
    return std::min(1.0, std::min(implicitBarrier.InitStepSize(solverData, p), barrier.InitStepSize(solverData, p, XTmp)));
}

int IPEnergy::NNZ(const SolverData<double>& solverData) const
{
    return inertia.NNZ(solverData) + implicitBarrier.NNZ(solverData) + elastic->NNZ(solverData) + barrier.NNZ(solverData);
}