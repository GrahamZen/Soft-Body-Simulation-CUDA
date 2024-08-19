#include <utilities.cuh>
#include <collision/aabb.h>
#include <collision/rigid/sphere.h>
#include <collision/rigid/plane.h>
#include <collision/rigid/cylinder.h>

__global__ void UpdateParticles(glm::vec3* X, glm::vec3* V, const glm::vec3* Force,
    int numVerts, float mass, float dt, float damp,
    glm::vec3 floorPos, glm::vec3 floorUp, float muT, float muN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVerts) return;

    V[i] += Force[i] / mass * dt;
    V[i] *= damp;
    X[i] += V[i] * dt;

    float signedDis = glm::dot(X[i] - floorPos, floorUp);
    if (signedDis < 0 && glm::dot(V[i], floorUp) < 0) {
        X[i] -= signedDis * floorUp;
        glm::vec3 vN = glm::dot(V[i], floorUp) * floorUp;
        glm::vec3 vT = V[i] - vN;
        float mag_vT = glm::length(vT);
        float a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(vN) / mag_vT, 0.0f);
        V[i] = -muN * vN + a * vT;
    }
}

__global__ void handleFloorCollision(glm::vec3* X, glm::vec3* V, int numVerts, Plane* planes, int numPlanes, float muT, float muN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVerts) return;
    V[i] *= 0.99f;
    for (int j = 0; j < numPlanes; j++)
    {
        glm::vec3 floorPos = glm::vec3(planes[j].m_model[3]);
        glm::vec3 floorUp = planes[j].m_floorUp;
        float signedDis = glm::dot(X[i] - floorPos, floorUp);
        if (signedDis < 0 && glm::dot(V[i], floorUp) < 0) {
            X[i] -= signedDis * floorUp;
            glm::vec3 vN = glm::dot(V[i], floorUp) * floorUp;
            glm::vec3 vT = V[i] - vN;
            float mag_vT = glm::length(vT);
            float a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(vN) / mag_vT, 0.0f);
            V[i] = -muN * vN + a * vT;
        }
    }
}


__global__ void handleSphereCollision(glm::vec3* X, glm::vec3* V, int numVerts, Sphere* spheres, int numSpheres, float muT, float muN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVerts) return;

    for (int j = 0; j < numSpheres; j++) {
        glm::vec3 sphereCenter = glm::vec3(spheres[j].m_model[3]);
        float sphereRadius = spheres[j].m_radius;
        glm::vec3 toCenter = X[i] - sphereCenter;
        float distance = glm::length(toCenter);
        if (distance < sphereRadius) {
            glm::vec3 normal = glm::normalize(toCenter);
            X[i] += distance * normal;
            glm::vec3 vN = glm::dot(V[i], normal) * normal;
            glm::vec3 vT = V[i] - vN;
            float mag_vT = glm::length(vT);
            float a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(vN) / mag_vT, 0.0f);
            V[i] = -muN * vN + a * vT;
        }
    }
}

__global__ void handleCylinderCollision(glm::vec3* X, glm::vec3* V, int numVerts, Cylinder* cylinders, int numCylinders, float muT, float muN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVerts) return;

    for (int j = 0; j < numCylinders; j++) {
        const Cylinder cy = cylinders[j];
        glm::vec3 axis = glm::vec3(glm::normalize(cy.m_model * glm::vec4(0.f, 1.f, 0.f, 0.f)));
        glm::vec3 cylinderCenter = glm::vec3(cylinders[j].m_model[3]);
        float cylinderRadius = cylinders[j].m_radius;

        glm::vec3 toCenter = X[i] - cylinderCenter;
        glm::vec3 projOnAxis = glm::dot(toCenter, axis) * axis;
        glm::vec3 closestPoint = cylinderCenter + projOnAxis;
        glm::vec3 toClosestPoint = X[i] - closestPoint;
        float distance = glm::length(toClosestPoint);

        if (distance < cylinderRadius) {
            glm::vec3 normal = glm::normalize(toClosestPoint);
            X[i] += (cylinderRadius - distance) * normal;
            glm::vec3 vN = glm::dot(V[i], normal) * normal;
            glm::vec3 vT = V[i] - vN;
            float mag_vT = glm::length(vT);
            float a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(vN) / mag_vT, 0.0f);
            V[i] = -muN * vN + a * vT;
        }
    }
}
