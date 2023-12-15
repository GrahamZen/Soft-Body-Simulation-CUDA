#include <simulation/solver/projective/pdSolver.h>
#include <simulation/solver/explicit/explicitSolver.h>
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
    cudaMalloc((void**)&solverData.V0, sizeof(float) * solverData.numTets);
    cudaMemset(solverData.V0, 0, sizeof(float) * solverData.numTets);
    cudaMalloc((void**)&solverData.inv_Dm, sizeof(glm::mat4) * solverData.numTets);

    int blocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    computeInvDmV0 << < blocks, threadsPerBlock >> > (solverData.V0, solverData.inv_Dm, solverData.numTets, solverData.X, solverData.Tet);
}

PdSolver::~PdSolver() {
    cudaFree(sn);
    cudaFree(b);
    cudaFree(masses);

    free(bHost);
    cudaFree(buffer_gpu);
}

ExplicitSolver::ExplicitSolver(SimulationCUDAContext* context, const SolverData& solverData) : FEMSolver(context)
{
    if (!solverData.dev_ExtForce)
        cudaMalloc((void**)&solverData.dev_ExtForce, sizeof(glm::vec3) * solverData.numVerts);
    cudaMemset(solverData.dev_ExtForce, 0, sizeof(glm::vec3) * solverData.numVerts);
    if (!solverData.inv_Dm)
        cudaMalloc((void**)&solverData.inv_Dm, sizeof(glm::mat4) * solverData.numTets);
    if (!solverData.V0)
        cudaMalloc((void**)&solverData.V0, sizeof(float) * solverData.numTets);
    cudaMemset(solverData.V0, 0, sizeof(float) * solverData.numTets);
    cudaMalloc((void**)&V_sum, sizeof(glm::vec3) * solverData.numVerts);
    cudaMalloc((void**)&V_num, sizeof(int) * solverData.numVerts);
    cudaMemset(V_sum, 0, sizeof(glm::vec3) * solverData.numVerts);
    cudaMemset(V_num, 0, sizeof(int) * solverData.numVerts);
    int blocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    computeInvDmV0 << < blocks, threadsPerBlock >> > (solverData.V0, solverData.inv_Dm, solverData.numTets, solverData.X, solverData.Tet);
}

ExplicitSolver::~ExplicitSolver()
{
}
