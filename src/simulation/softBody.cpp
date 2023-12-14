#include <softBody.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <glm/glm.hpp>
#include <bvh.cuh>
#include <simulationContext.h>

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
    //solver->setAttributes(softBodyAttr);
}
