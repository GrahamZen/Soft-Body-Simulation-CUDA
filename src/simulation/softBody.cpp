#include <simulation/softBody.h>
#include <simulation/simulationContext.h>
#include <context.h>

SoftBody::SoftBody(const SoftBodyData* dataPtr, const SoftBodyAttribute _attrib, int threadsPerBlock) :softBodyData(*dataPtr), attrib(_attrib), threadsPerBlock(threadsPerBlock)
{
    Mesh::numTris = softBodyData.numTris;
    createMesh();
}

int SoftBody::GetNumTris() const
{
    return numTris;
}
const SoftBodyData& SoftBody::GetSoftBodyData()const
{
    return softBodyData;
}

void SoftBody::SetAttributes(SoftBodyAttr* pSoftBodyAttr)
{
    pSoftBodyAttr->setJumpClean(attrib.jump);
    if (pSoftBodyAttr->mu.second)
        attrib.mu = pSoftBodyAttr->mu.first;
    if (pSoftBodyAttr->lambda.second)
        attrib.lambda = pSoftBodyAttr->lambda.first;
}
