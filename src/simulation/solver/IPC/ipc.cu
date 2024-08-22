#include <IPC/ipc.h>
#include <cuda_runtime.h>

IPCSolver::IPCSolver(int threadsPerBlock, const SolverData& solverData) :FEMSolver(threadsPerBlock), numVerts(solverData.numVerts)
{
    cudaMalloc(&gradient, sizeof(double) * numVerts * 3);
}

IPCSolver::~IPCSolver()
{
    cudaFree(gradient);
}

void IPCSolver::Update(SolverData& solverData, SolverParams& solverParams)
{
}

void IPCSolver::SolverPrepare(SolverData& solverData, SolverParams& solverParams)
{
}

void IPCSolver::SolverStep(SolverData& solverData, SolverParams& solverParams)
{
}
