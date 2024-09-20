#include <fixedBodyData.h>
#include <collision/rigid/sphere.h>
#include <collision/rigid/plane.h>
#include <collision/rigid/cylinder.h>

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


template<typename Scalar>
__global__ void handleFloorCollision(glm::tvec3<Scalar>* X, glm::tvec3<Scalar>* V, int numVerts, Plane* planes, int numPlanes, Scalar muT, Scalar muN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVerts) return;
    for (int j = 0; j < numPlanes; j++)
    {
        glm::tvec3<Scalar> floorPos = glm::tvec3<Scalar>(planes[j].m_model[3]);
        glm::tvec3<Scalar> floorUp = planes[j].m_floorUp;
        Scalar signedDis = glm::dot(X[i] - floorPos, floorUp);
        if (signedDis < 0 && glm::dot(V[i], floorUp) < 0) {
            X[i] -= signedDis * floorUp;
            glm::tvec3<Scalar> vN = glm::dot(V[i], floorUp) * floorUp;
            glm::tvec3<Scalar> vT = V[i] - vN;
            Scalar mag_vT = glm::length(vT);
            Scalar a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(vN) / mag_vT, (Scalar)0);
            V[i] = -muN * vN + a * vT;
        }
    }
}


template<typename Scalar>
__global__ void handleSphereCollision(glm::tvec3<Scalar>* X, glm::tvec3<Scalar>* V, int numVerts, Sphere* spheres, int numSpheres, Scalar muT, Scalar muN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVerts) return;

    for (int j = 0; j < numSpheres; j++) {
        glm::tvec3<Scalar> sphereCenter = glm::tvec3<Scalar>(spheres[j].m_model[3]);
        Scalar sphereRadius = spheres[j].m_radius;
        glm::tvec3<Scalar> toCenter = X[i] - sphereCenter;
        Scalar d = glm::length(toCenter);
        if (d < sphereRadius) {
            glm::tvec3<Scalar> normal = glm::normalize(toCenter);
            X[i] += d * normal;
            glm::tvec3<Scalar> vN = glm::dot(V[i], normal) * normal;
            glm::tvec3<Scalar> vT = V[i] - vN;
            Scalar mag_vT = glm::length(vT);
            Scalar a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(vN) / mag_vT, (Scalar)0);
            V[i] = -muN * vN + a * vT;
        }
    }
}

template<typename Scalar>
__global__ void handleCylinderCollision(glm::tvec3<Scalar>* X, glm::tvec3<Scalar>* V, int numVerts, Cylinder* cylinders, int numCylinders, Scalar muT, Scalar muN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVerts) return;

    for (int j = 0; j < numCylinders; j++) {
        const Cylinder cy = cylinders[j];
        glm::tvec3<Scalar> axis = glm::tvec3<Scalar>(glm::normalize(cy.m_model * glm::vec4(0.f, 1.f, 0.f, 0.f)));
        glm::tmat3x3<Scalar> nnT =glm::tmat3x3<Scalar>(1.f) - glm::outerProduct(axis, axis);
        glm::tvec3<Scalar> cylinderCenter = glm::tvec3<Scalar>(cy.m_model[3]);
        Scalar cylinderRadius = cy.m_radius;
        glm::tvec3<Scalar> n = nnT * (X[i] - cylinderCenter);
        Scalar d = glm::length(n);

        if (d < cylinderRadius) {
            glm::tvec3<Scalar> normal = glm::normalize(n);
            X[i] += (cylinderRadius - d) * normal;
            glm::tvec3<Scalar> vN = glm::dot(V[i], normal) * normal;
            glm::tvec3<Scalar> vT = V[i] - vN;
            Scalar mag_vT = glm::length(vT);
            Scalar a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(vN) / mag_vT, (Scalar)0);
            V[i] = -muN * vN + a * vT;
        }
    }
}

template<typename Scalar>
void FixedBodyData::HandleCollisions<Scalar>(glm::tvec3<Scalar>* X, glm::tvec3<Scalar>* V, int numVerts, Scalar muT, Scalar muN) {
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