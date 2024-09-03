#pragma once

#include <rigid.h>
#include <vector>

class Sphere;
class Plane;
class Cylinder;

class FixedBodyData {
    template<typename Scalar>
    friend class BarrierEnergy;
public:
    FixedBodyData();
    FixedBodyData(int threadsPerBlock, const std::vector<FixedBody*>&);
    FixedBodyData(const FixedBodyData&) = delete;
    FixedBodyData& operator=(const FixedBodyData&) = delete;
    FixedBodyData(FixedBodyData&&) = default;
    FixedBodyData& operator=(FixedBodyData&&) = default;
    template<typename Scalar>
    void HandleCollisions(glm::tvec3<Scalar>* X, glm::tvec3<Scalar>* V, int numVerts, Scalar muT, Scalar muN);
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