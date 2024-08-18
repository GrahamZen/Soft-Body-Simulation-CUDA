#include <simulation/softBody.h>

SoftBody::~SoftBody()
{
    cudaFree(softBodyData.Tri);
}

void SoftBody::Reset()
{
}
