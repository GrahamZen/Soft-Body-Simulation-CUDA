#include <fixedBodyData.h>
#include <sphere.h>
#include <plane.h>
#include <collision/rigid/cylinder.h>
#include <utilities.cuh>

FixedBodyData::FixedBodyData() {}

FixedBodyData::FixedBodyData(int _threadsPerBlock, const std::vector<FixedBody*>& fixedBodies) : threadsPerBlock(_threadsPerBlock) {
    numSpheres = 0;
    numPlanes = 0;
    numCylinders = 0;
    for (auto fixedBody : fixedBodies) {
        switch (fixedBody->getType())
        {
        case BodyType::Sphere:
            numSpheres++;
            break;
        case BodyType::Plane:
            numPlanes++;
            break;
        case BodyType::Cylinder:
            numCylinders++;
        default:
            break;
        }
    }
    if (numSpheres > 0) {
        cudaMalloc(&dev_spheres, numSpheres * sizeof(Sphere));
    }
    if (numPlanes > 0) {
        cudaMalloc(&dev_planes, numPlanes * sizeof(Plane));
    }
    if (numCylinders > 0) {
        cudaMalloc(&dev_cylinders, numCylinders * sizeof(Cylinder));
    }
    int sphereIdx = 0;
    int floorIdx = 0;
    int cylinderIdx = 0;
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
        case BodyType::Cylinder:
            cudaMemcpy(dev_cylinders + cylinderIdx, (Cylinder*)fixedBody, sizeof(Cylinder), cudaMemcpyHostToDevice);
            cylinderIdx++;
            break;
        default:
            break;
        }
    }
}

FixedBodyData::~FixedBodyData() {
    cudaFree(dev_spheres);
    cudaFree(dev_planes);
    cudaFree(dev_cylinders);
}


template<typename HighP>
__global__ void handleFloorCollision(glm::tvec3<HighP>* X, glm::tvec3<HighP>* V, int numVerts, Plane* planes, int numPlanes, HighP muT, HighP muN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVerts) return;
    for (int j = 0; j < numPlanes; j++)
    {
        glm::tvec3<HighP> floorPos = glm::tvec3<HighP>(planes[j].m_model[3]);
        glm::tvec3<HighP> floorUp = planes[j].m_floorUp;
        HighP signedDis = glm::dot(X[i] - floorPos, floorUp);
        if (signedDis < 0 && glm::dot(V[i], floorUp) < 0) {
            X[i] -= signedDis * floorUp;
            glm::tvec3<HighP> vN = glm::dot(V[i], floorUp) * floorUp;
            glm::tvec3<HighP> vT = V[i] - vN;
            HighP mag_vT = glm::length(vT);
            HighP a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(vN) / mag_vT, (HighP)0);
            V[i] = -muN * vN + a * vT;
        }
    }
}


template<typename HighP>
__global__ void handleSphereCollision(glm::tvec3<HighP>* X, glm::tvec3<HighP>* V, int numVerts, Sphere* spheres, int numSpheres, HighP muT, HighP muN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVerts) return;

    for (int j = 0; j < numSpheres; j++) {
        glm::tvec3<HighP> sphereCenter = glm::tvec3<HighP>(spheres[j].m_model[3]);
        HighP sphereRadius = spheres[j].m_radius;
        glm::tvec3<HighP> toCenter = X[i] - sphereCenter;
        HighP distance = glm::length(toCenter);
        if (distance < sphereRadius) {
            glm::tvec3<HighP> normal = glm::normalize(toCenter);
            X[i] += distance * normal;
            glm::tvec3<HighP> vN = glm::dot(V[i], normal) * normal;
            glm::tvec3<HighP> vT = V[i] - vN;
            HighP mag_vT = glm::length(vT);
            HighP a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(vN) / mag_vT, (HighP)0);
            V[i] = -muN * vN + a * vT;
        }
    }
}

template<typename HighP>
__global__ void handleCylinderCollision(glm::tvec3<HighP>* X, glm::tvec3<HighP>* V, int numVerts, Cylinder* cylinders, int numCylinders, HighP muT, HighP muN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVerts) return;

    for (int j = 0; j < numCylinders; j++) {
        const Cylinder cy = cylinders[j];
        glm::tvec3<HighP> axis = glm::tvec3<HighP>(glm::normalize(cy.m_model * glm::vec4(0.f, 1.f, 0.f, 0.f)));
        glm::tvec3<HighP> cylinderCenter = glm::tvec3<HighP>(cylinders[j].m_model[3]);
        HighP cylinderRadius = cylinders[j].m_radius;

        glm::tvec3<HighP> toCenter = X[i] - cylinderCenter;
        glm::tvec3<HighP> projOnAxis = glm::dot(toCenter, axis) * axis;
        glm::tvec3<HighP> closestPoint = cylinderCenter + projOnAxis;
        glm::tvec3<HighP> toClosestPoint = X[i] - closestPoint;
        HighP distance = glm::length(toClosestPoint);

        if (distance < cylinderRadius) {
            glm::tvec3<HighP> normal = glm::normalize(toClosestPoint);
            X[i] += (cylinderRadius - distance) * normal;
            glm::tvec3<HighP> vN = glm::dot(V[i], normal) * normal;
            glm::tvec3<HighP> vT = V[i] - vN;
            HighP mag_vT = glm::length(vT);
            HighP a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(vN) / mag_vT, (HighP)0);
            V[i] = -muN * vN + a * vT;
        }
    }
}

template<typename HighP>
void FixedBodyData::HandleCollisions<HighP>(glm::tvec3<HighP>* X, glm::tvec3<HighP>* V, int numVerts, HighP muT, HighP muN) {
    int numBlocks = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    if (numSpheres > 0) {
        handleSphereCollision << <numBlocks, threadsPerBlock >> > (X, V, numVerts, dev_spheres, numSpheres, muT, muN);
    }
    if (numPlanes > 0) {
        handleFloorCollision << <numBlocks, threadsPerBlock >> > (X, V, numVerts, dev_planes, numPlanes, muT, muN);
    }
    if (numCylinders > 0) {
        handleCylinderCollision << <numBlocks, threadsPerBlock >> > (X, V, numVerts, dev_cylinders, numCylinders, muT, muN);
    }
}

template void FixedBodyData::HandleCollisions<float>(glm::vec3*, glm::vec3*, int, float, float);
template void FixedBodyData::HandleCollisions<double>(glm::tvec3<double>*, glm::tvec3<double>*, int, double, double);