#include <fixedBodyData.h>
#include <sphere.h>
#include <plane.h>
#include <utilities.cuh>

FixedBodyData::FixedBodyData() {}

FixedBodyData::FixedBodyData(int _threadsPerBlock, const std::vector<FixedBody*>& fixedBodies) : threadsPerBlock(_threadsPerBlock) {
    numSpheres = 0;
    numPlanes = 0;
    for (auto fixedBody : fixedBodies) {
        switch (fixedBody->getType())
        {
        case BodyType::Sphere:
            numSpheres++;
            break;
        case BodyType::Plane:
            numPlanes++;
            break;
        default:
            break;
        }
    }
    if (numSpheres > 0) {
        cudaMalloc(&dev_spheres, numSpheres * sizeof(Sphere));
    }
    if (numSpheres > 0) {
        cudaMalloc(&dev_planes, numPlanes * sizeof(Plane));
    }
    int sphereIdx = 0;
    int floorIdx = 0;
    for (auto fixedBody : fixedBodies) {
        switch (fixedBody->getType())
        {
        case BodyType::Sphere:
            cudaMemcpy(dev_spheres + sphereIdx, (Sphere*)fixedBody, sizeof(Sphere), cudaMemcpyHostToDevice);
            sphereIdx++;
            break;
        case BodyType::Plane:
            cudaMemcpy(dev_planes + floorIdx, (Plane*)fixedBody, sizeof(Plane), cudaMemcpyHostToDevice);
            floorIdx++;
            break;
        default:
            break;
        }
    }
}

FixedBodyData::~FixedBodyData() {
    cudaFree(dev_spheres);
}

void FixedBodyData::HandleCollisions(glm::vec3* X, glm::vec3* V, int numVerts, float muT, float muN) {
    int numBlocks = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    if (numSpheres > 0) {
        handleSphereCollision << <numBlocks, threadsPerBlock >> > (X, V, numVerts, dev_spheres, numSpheres, muT, muN);
    }
    if (numPlanes > 0) {
        handleFloorCollision << <numBlocks, threadsPerBlock >> > (X, V, numVerts, dev_planes, numPlanes, muT, muN);
    }
}
