#pragma once

#include <def.h>

namespace ExplicitUtil
{
    __global__ void ComputeForcesSVD(glm::vec3* Force, const glm::vec3* X, const indexType* Tet, int tet_number, const glm::mat3* DmInv, float mu, float lambda);
    __global__ void EulerMethod(glm::vec3* X, glm::vec3* V, const glm::vec3* Force, int numVerts, float mass, float dt);
    __global__ void LaplacianKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int numVerts, const indexType* Tet, float blendAlpha);
    __global__ void LaplacianGatherKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int numTets, const indexType* Tet);
}
