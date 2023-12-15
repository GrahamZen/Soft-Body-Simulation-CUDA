#include <simulation/softBody.h>
#include <simulation/solver/projective/pdSolver.h>
#include <simulation/solver/explicit/explicitSolver.h>
#include <simulation/simulationContext.h>

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

void SoftBody::Update()
{
    solver->Update(solverData, attrib);
}

int SoftBody::GetNumVerts() const
{
    return solverData.numVerts;
}

int SoftBody::GetNumTets() const
{
    return solverData.numTets;
}

int SoftBody::GetNumTris() const
{
    return numTris;
}
const SolverData& SoftBody::GetSolverData()const
{
    return solverData;
}

void SoftBody::SetAttributes(GuiDataContainer::SoftBodyAttr& softBodyAttr)
{
    softBodyAttr.setJumpClean(attrib.jump);
    if (softBodyAttr.stiffness_0.second)
        attrib.stiffness_0 = softBodyAttr.stiffness_0.first;
    if (softBodyAttr.stiffness_1.second)
        attrib.stiffness_1 = softBodyAttr.stiffness_1.first;
}
