#include <softBody.h>
#include <simulation/simulationContext.h>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <utilities.cuh>
#include <pdSolver.h>

SoftBody::SoftBody(SimulationCUDAContext* context, SolverAttribute& _attrib, SolverData* dataPtr)
    :solverData(*dataPtr), attrib(_attrib), solver(new PdSolver{ context, solverData }), threadsPerBlock(context->GetThreadsPerBlock())
{
    Mesh::numTets = solverData.numTets;
    Mesh::numTris = solverData.numTris;
    if (numTris == 0)
        createTetrahedron();
    else
        createMesh();
}

SoftBody::~SoftBody()
{
    cudaFree(solverData.Tet);
    cudaFree(solverData.Force);
    cudaFree(solverData.V);
    cudaFree(solverData.inv_Dm);

    delete solver;
}

void SoftBody::Update()
{
    solver->Update(solverData, attrib);
}

void SoftBody::Reset()
{
    cudaMemcpy(solverData.X, solverData.X0, sizeof(glm::vec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
    cudaMemcpy(solverData.XTilt, solverData.X0, sizeof(glm::vec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
    cudaMemset(solverData.V, 0, sizeof(glm::vec3) * solverData.numVerts);
}
