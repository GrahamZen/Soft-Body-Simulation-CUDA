#include <simulation/softBody.h>

SoftBody::~SoftBody()
{
    cudaFree(softBodyData.Tri);
}
