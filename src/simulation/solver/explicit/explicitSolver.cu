#include <simulation/solver/explicit/explicitSolver.h>
#include <simulation/solver/explicit/explicitUtil.cuh>
#include <simulation/solver/solverUtil.cuh>
#include <simulation/simulationContext.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

ExplicitSolver::ExplicitSolver(int threadsPerBlock, const SolverData<float>& solverData) : FEMSolver(threadsPerBlock)
{
    if (!solverData.dev_ExtForce)
        cudaMalloc((void**)&solverData.dev_ExtForce, sizeof(glm::vec3) * solverData.numVerts);
    cudaMemset(solverData.dev_ExtForce, 0, sizeof(glm::vec3) * solverData.numVerts);
    if (!solverData.inv_Dm)
        cudaMalloc((void**)&solverData.inv_Dm, sizeof(glm::mat4) * solverData.numTets);
    cudaMalloc((void**)&V_sum, sizeof(glm::vec3) * solverData.numVerts);
    cudaMalloc((void**)&V_num, sizeof(int) * solverData.numVerts);
    cudaMemset(V_sum, 0, sizeof(glm::vec3) * solverData.numVerts);
    cudaMemset(V_num, 0, sizeof(int) * solverData.numVerts);
    int blocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    ExplicitUtil::computeInvDm << < blocks, threadsPerBlock >> > (solverData.inv_Dm, solverData.numTets, solverData.X, solverData.Tet);
}

ExplicitSolver::~ExplicitSolver()
{
}


void ExplicitSolver::SolverPrepare(SolverData<float>& solverData, SolverParams& solverParams)
{
}


void ExplicitSolver::SolverStep(SolverData<float>& solverData, SolverParams& solverParams)
{
    glm::vec3 gravity{ 0.0f, -solverParams.gravity * solverParams.solverAttr.mass, 0.0f };
    thrust::device_ptr<glm::vec3> dev_ptr(solverData.Force);
    thrust::fill(thrust::device, dev_ptr, dev_ptr + solverData.numVerts, gravity);
    Laplacian_Smoothing(solverData, 0.5);
    ExplicitUtil::ComputeForcesSVD << <(solverData.numTets + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (solverData.Force, solverData.XTilde, solverData.Tet, solverData.numTets, solverData.inv_Dm, solverParams.solverAttr.stiffness_0, solverParams.solverAttr.stiffness_1);
    ExplicitUtil::EulerMethod << <(solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (solverData.XTilde, solverData.V, solverData.Force, solverData.numVerts, solverParams.solverAttr.mass, solverParams.dt);
}


void ExplicitSolver::Update(SolverData<float>& solverData, SolverParams& solverParams)
{
    AddExternal << <(solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (solverData.V, solverData.numVerts, solverParams.solverAttr.jump, solverParams.solverAttr.mass, solverParams.extForce.jump);
    for (size_t i = 0; i < 10; i++)
    {
        SolverStep(solverData, solverParams);
    }
}


void ExplicitSolver::Laplacian_Smoothing(SolverData<float>& solverData, float blendAlpha)
{
    cudaMemset(V_sum, 0, sizeof(glm::vec3) * solverData.numVerts);
    cudaMemset(V_num, 0, sizeof(int) * solverData.numVerts);
    int blocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    ExplicitUtil::LaplacianGatherKern << < blocks, threadsPerBlock >> > (solverData.V, V_sum, V_num, solverData.numTets, solverData.Tet);
    ExplicitUtil::LaplacianKern << < (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (solverData.V, V_sum, V_num, solverData.numVerts, solverData.Tet, blendAlpha);
}