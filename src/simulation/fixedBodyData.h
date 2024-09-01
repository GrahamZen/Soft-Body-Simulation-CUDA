#pragma once

#include <rigid.h>
#include <vector>

class Sphere;
class Plane;
class Cylinder;

class FixedBodyData {
    template<typename HighP>
    friend class BarrierEnergy;
public:
    FixedBodyData();
    FixedBodyData(int threadsPerBlock, const std::vector<FixedBody*>&);
    FixedBodyData(const FixedBodyData&) = delete;
    FixedBodyData& operator=(const FixedBodyData&) = delete;
    FixedBodyData(FixedBodyData&&) = default;
    FixedBodyData& operator=(FixedBodyData&&) = default;
    template<typename HighP>
    void HandleCollisions(glm::tvec3<HighP>* X, glm::tvec3<HighP>* V, int numVerts, HighP muT, HighP muN);
    ~FixedBodyData();
private:
    Sphere* dev_spheres = nullptr;
    Plane* dev_planes = nullptr;
    Cylinder* dev_cylinders = nullptr;
    int numSpheres;
    int numPlanes;
    int numCylinders;
    int threadsPerBlock;
};