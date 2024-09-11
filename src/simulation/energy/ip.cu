#include <energy/ip.h>
#include <energy/corotated.h>
#include <bvh.h>

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
    return inertia.Val(Xs, solverData) + h2 * (gravity.Val(Xs, solverData) + elastic->Val(Xs, solverData) + implicitBarrier.Val(Xs, solverData));
}

void IPEnergy::Gradient(const SolverData<double>& solverData, double h2) const
{
    cudaMemset(gradient, 0, sizeof(double) * solverData.numVerts * 3);
    inertia.Gradient(gradient, solverData, 1);
    gravity.Gradient(gradient, solverData, h2);
    elastic->Gradient(gradient, solverData, h2);
    implicitBarrier.Gradient(gradient, solverData, h2);
}

void IPEnergy::Hessian(const SolverData<double>& solverData, double h2) const
{
    cudaMemset(hessianVal, 0, sizeof(double) * nnz);
    inertia.Hessian(solverData, 1);
    gravity.Hessian(solverData, h2);
    elastic->Hessian(solverData, h2);
    implicitBarrier.Hessian(solverData, h2);
}

double IPEnergy::InitStepSize(SolverData<double>& solverData, double* p, glm::dvec3* XTmp) const
{
    double alpha = solverData.pCollisionDetection->ComputeMinStepSize(solverData.numVerts, solverData.numTris, solverData.Tri, solverData.X, XTmp,
        solverData.dev_TriFathers, true);
    return std::min(alpha, implicitBarrier.InitStepSize(solverData, p));
}

void IPEnergy::UpdateQueries(CollisionDetection<double>* cd, int numVerts, int numTris, const indexType* Tri, const glm::tvec3<double>* X, const indexType* TriFathers, Query*& queries, int& _numQueries)
{
    cd->BroadPhase(numVerts, numTris, Tri, X, TriFathers, queries, _numQueries);
}