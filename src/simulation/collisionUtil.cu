#include <def.h>


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
        float a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(vN) / mag_vT, (float) 0);
        V[i] = -muN * vN + a * vT;
    }
}


template <typename Scalar>
__global__ void IPCCDKernel(glm::tvec3<Scalar>* X, glm::tvec3<Scalar>* XTilde, glm::tvec3<Scalar>* V, Scalar* tI, glm::vec3* normals, float muT, float muN, int numVerts) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numVerts) return;
    Scalar interval = glm::length(XTilde - X);

    if (tI[idx] < 1.0f)
    {
        glm::tvec3<Scalar> normal = normals[idx];
        glm::tvec3<Scalar> vel = XTilde[idx] - X[idx];
        glm::tvec3<Scalar> velNormal = glm::dot(vel, normal) * normal;
        glm::tvec3<Scalar> vT = vel - velNormal;
        Scalar mag_vT = glm::length(vT);
        //Scalar a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(velNormal) / mag_vT, 0.0);
        V[idx] = (Scalar)-muN * velNormal;
        // V[idx] = X[idx] - XTilde[idx];
    }
    else
    {
        X[idx] = XTilde[idx];
    }
    //XTilde[idx] = X[idx];
}

template <typename Scalar>
__global__ void CCDKernel(glm::tvec3<Scalar>* X, glm::tvec3<Scalar>* XTilde, glm::tvec3<Scalar>* V, Scalar* tI, glm::vec3* normals, float muT, float muN, int numVerts, Scalar dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numVerts) return;
    Scalar interval = glm::length(XTilde - X);

    if (tI[idx] < 1.0f)
    {
        glm::tvec3<Scalar> normal = normals[idx];
        glm::tvec3<Scalar> vel = XTilde[idx] - X[idx];
        glm::tvec3<Scalar> velNormal = glm::dot(vel, normal) * normal;
        glm::tvec3<Scalar> vT = vel - velNormal;
        Scalar mag_vT = glm::length(vT);
        //Scalar a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(velNormal) / mag_vT, 0.0);
        V[idx] = -velNormal;
        // V[idx] = X[idx] - XTilde[idx];
    }
    else
    {
        X[idx] = XTilde[idx];
    }
    //XTilde[idx] = X[idx];
}

template __global__ void IPCCDKernel<float>(glm::tvec3<float>* X, glm::tvec3<float>* XTilde, glm::tvec3<float>* V, float* tI, glm::vec3* normals, float muT, float muN, int numVerts);
template __global__ void IPCCDKernel<double>(glm::tvec3<double>* X, glm::tvec3<double>* XTilde, glm::tvec3<double>* V, double* tI, glm::vec3* normals, float muT, float muN, int numVerts);
template __global__ void CCDKernel<float>(glm::tvec3<float>* X, glm::tvec3<float>* XTilde, glm::tvec3<float>* V, float* tI, glm::vec3* normals, float muT, float muN, int numVerts, float dt);
template __global__ void CCDKernel<double>(glm::tvec3<double>* X, glm::tvec3<double>* XTilde, glm::tvec3<double>* V, double* tI, glm::vec3* normals, float muT, float muN, int numVerts, double dt);

