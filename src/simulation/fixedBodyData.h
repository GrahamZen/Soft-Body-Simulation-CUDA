#pragma once

#include <rigid.h>
#include <vector>

class Sphere;
class Plane;

class FixedBodyData {
public:
    FixedBodyData();
    FixedBodyData(int threadsPerBlock, const std::vector<FixedBody*>&);
    FixedBodyData(const FixedBodyData&) = delete;
    FixedBodyData& operator=(const FixedBodyData&) = delete;
    FixedBodyData(FixedBodyData&&) = default;
    FixedBodyData& operator=(FixedBodyData&&) = default;
    void HandleCollisions(glm::vec3* X, glm::vec3* V, int numVerts, float muT, float muN);
    ~FixedBodyData();
private:
    Sphere* dev_spheres = nullptr;
    Plane* dev_planes = nullptr;
    int numSpheres;
    int numPlanes;
    int threadsPerBlock;
};