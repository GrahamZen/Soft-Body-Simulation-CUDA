#include <IPC/ipc.h>
#include <cuda_runtime.h>

IPCSolver::IPCSolver(int threadsPerBlock, const SolverData<double>& solverData)
    :FEMSolver(threadsPerBlock), numVerts(solverData.numVerts), inertia(numVerts, solverData.mass)
{
    cudaMalloc(&gradient, sizeof(double) * numVerts * 3);
}

IPCSolver::~IPCSolver()
{
    cudaFree(gradient);
}

void IPCSolver::Update(SolverData<double>& solverData, SolverParams& solverParams)
{
}

void IPCSolver::SolverPrepare(SolverData<double>& solverData, SolverParams& solverParams)
{
}

void IPCSolver::SolverStep(SolverData<double>& solverData, SolverParams& solverParams)
{
}
