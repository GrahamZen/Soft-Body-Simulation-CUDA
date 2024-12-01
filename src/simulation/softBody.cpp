#include <simulation/softBody.h>
#include <simulation/simulationContext.h>
#include <context.h>

SoftBody::SoftBody(const SoftBodyData* dataPtr, const SoftBodyAttribute _attrib, std::pair<size_t, size_t> _tetIdxRange, int threadsPerBlock, const char* _name)
    :softBodyData(*dataPtr), attrib(_attrib), threadsPerBlock(threadsPerBlock), name(_name), tetIdxRange(_tetIdxRange)
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