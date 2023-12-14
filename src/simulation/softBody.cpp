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
    softBodyAttr.setJumpClean(attrib.jump);
    if (softBodyAttr.stiffness_0.second)
        attrib.stiffness_0 = softBodyAttr.stiffness_0.first;
    if (softBodyAttr.stiffness_1.second)
        attrib.stiffness_1 = softBodyAttr.stiffness_1.first;
}
