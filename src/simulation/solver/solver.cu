#include <simulation/solver/pdSolver.h>
#include <simulation/simulationContext.h>
#include <utilities.cuh>

Solver::Solver(SimulationCUDAContext* context) :mcrpSimContext(context), threadsPerBlock(context->GetThreadsPerBlock())
{

}

Solver::~Solver() {
}

FEMSolver::FEMSolver(SimulationCUDAContext* context) : Solver(context) {}

PdSolver::PdSolver(SimulationCUDAContext* context, const SolverData& solverData) : FEMSolver(context)
{
    cudaMalloc((void**)&solverData.dev_ExtForce, sizeof(glm::vec3) * solverData.numVerts);
    cudaMemset(solverData.dev_ExtForce, 0, sizeof(glm::vec3) * solverData.numVerts);
    cudaMalloc((void**)&V0, sizeof(float) * solverData.numTets);
    cudaMemset(V0, 0, sizeof(float) * solverData.numTets);
    cudaMalloc((void**)&solverData.inv_Dm, sizeof(glm::mat4) * solverData.numTets);

    int blocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    computeInvDmV0 << < blocks, threadsPerBlock >> > (V0, solverData.inv_Dm, solverData.numTets, solverData.X, solverData.Tet);
}

PdSolver::~PdSolver() {
    cudaFree(sn);
    cudaFree(b);
    cudaFree(masses);

    free(bHost);
    cudaFree(buffer_gpu);
}