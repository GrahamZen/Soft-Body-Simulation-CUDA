#include <simulation/softBody.h>
#include <simulation/simulationContext.h>

SoftBody::SoftBody(const SoftBodyData* dataPtr, const SoftBodyAttribute _attrib, int threadsPerBlock) :softBodyData(*dataPtr), attrib(_attrib), threadsPerBlock(threadsPerBlock)
{
    Mesh::numTets = softBodyData.numTets;
    Mesh::numTris = softBodyData.numTris;
    if (numTris == 0)
        createTetrahedron();
    else
        createMesh();
}

int SoftBody::GetNumTets() const
{
    return softBodyData.numTets;
}

int SoftBody::GetNumTris() const
{
    return numTris;
}
const SoftBodyData& SoftBody::GetSoftBodyData()const
{
    return softBodyData;
}

void SoftBody::SetAttributes(GuiDataContainer::SoftBodyAttr& softBodyAttr)
{
    softBodyAttr.setJumpClean(attrib.jump);
    if (softBodyAttr.mu.second)
        attrib.mu = softBodyAttr.mu.first;
    if (softBodyAttr.lambda.second)
        attrib.lambda = softBodyAttr.lambda.first;
}
