#include <simulation/solver/explicit/explicitSolver.h>
#include <utilities.cuh>
#include <simulation/simulationContext.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

__global__ void EulerMethod(glm::vec3* X, glm::vec3* V, const glm::vec3* Force, int numVerts, float mass, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVerts) return;

    V[i] += Force[i] / mass * dt;
    X[i] += V[i] * dt;
}

void ExplicitSolver::SolverPrepare(SolverData& solverData, SolverAttribute& attrib)
{
}


void ExplicitSolver::SolverStep(SolverData& solverData, SolverAttribute& attrib)
{
    glm::vec3 gravity{ 0.0f, -mcrpSimContext->GetGravity() * attrib.mass, 0.0f };
    thrust::device_ptr<glm::vec3> dev_ptr(solverData.Force);
    thrust::fill(thrust::device, dev_ptr, dev_ptr + solverData.numVerts, gravity);
    Laplacian_Smoothing(solverData, 0.5);
    ComputeForces << <(solverData.numTets + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (solverData.Force, solverData.X, solverData.Tet, solverData.numTets, solverData.inv_Dm, attrib.stiffness_0, attrib.stiffness_1);
    EulerMethod << <(solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (solverData.XTilt, solverData.V, solverData.Force, solverData.numVerts, attrib.mass, mcrpSimContext->GetDt());
}


void ExplicitSolver::Update(SolverData& solverData, SolverAttribute& attrib)
{
    AddExternal << <(solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (solverData.V, solverData.numVerts, attrib.jump, attrib.mass, mcrpSimContext->GetExtForce().jump);
    for (size_t i = 0; i < 50; i++)
    {
        SolverStep(solverData, attrib);
    }
}


void ExplicitSolver::Laplacian_Smoothing(SolverData& solverData, float blendAlpha)
{
    cudaMemset(V_sum, 0, sizeof(glm::vec3) * solverData.numVerts);
    cudaMemset(V_num, 0, sizeof(int) * solverData.numVerts);
    int blocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    LaplacianGatherKern << < blocks, threadsPerBlock >> > (solverData.V, V_sum, V_num, solverData.numTets, solverData.Tet);
    LaplacianKern << < (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (solverData.V, V_sum, V_num, solverData.numVerts, solverData.Tet, blendAlpha);
}