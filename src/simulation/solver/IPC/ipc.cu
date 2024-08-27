#include <IPC/ipc.h>
#include <energy/corotated.h>
#include <cuda_runtime.h>

IPCSolver::IPCSolver(int threadsPerBlock, const SolverData<double>& solverData)
    :FEMSolver(threadsPerBlock)
{
    cudaMalloc((void**)&solverData.V0, sizeof(float) * solverData.numTets);
    cudaMemset(solverData.V0, 0, sizeof(float) * solverData.numTets);
    cudaMalloc((void**)&solverData.DmInv, sizeof(glm::mat4) * solverData.numTets); 
    int blocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    computeInvDmV0 << < blocks, threadsPerBlock >> > (solverData.V0, solverData.DmInv, solverData.numTets, solverData.X, solverData.Tet);
 
    energies.push_back(new InertiaEnergy<double>(solverData, nnz, solverData.numVerts, solverData.mass));
    energies.push_back(new GravityEnergy<double>);
    energies.push_back(new CorotatedEnergy<double>(solverData, nnz));
    cudaMalloc(&gradient, sizeof(double) * solverData.numVerts * 3);
    cudaMalloc(&hessianVal, sizeof(double) * nnz);
    cudaMalloc(&hessianRowIdx, sizeof(int) * nnz);
    cudaMalloc(&hessianColIdx, sizeof(int) * nnz);
    for (int i = 0; i < energies.size(); i++)
    {
        energies[i]->SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
    }
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
    for (int i = 0; i < energies.size(); i++)
    {
        energies[i]->Gradient(gradient, solverData);
        energies[i]->Hessian(solverData);
    }
}
