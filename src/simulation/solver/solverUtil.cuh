#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

template<typename Scalar>
__forceinline__ __host__ __device__ void printGLMVector(const glm::tvec3<Scalar>& v, const char* name) {
    printf("Vector 3 %s\n%f %f %f \n--------------------------------\n", name, v.x, v.y, v.z);
}

template <typename Scalar>
__forceinline__ __host__ __device__ Scalar barrierFunc(Scalar d, Scalar dhat, float kappa, Scalar contact_area) {
    Scalar s = d / dhat;
    return contact_area * dhat * kappa * 0.5 * (s - 1) * log(s);
}

template <typename Scalar>
__forceinline__ __host__ __device__ glm::tvec3<Scalar> barrierFuncGrad(const glm::tvec3<Scalar>& normal, Scalar d, Scalar dhat, float kappa, Scalar contact_area) {
    Scalar s = d / dhat;
    return contact_area * dhat * (kappa / 2 * (log(s) / dhat + (s - 1) / d)) * normal;
}

template <typename Scalar>
__forceinline__ __host__ __device__ glm::tmat3x3<Scalar> barrierFuncHess(const glm::tvec3<Scalar>& normal, Scalar d, Scalar dhat, float kappa, Scalar contact_area) {
    return contact_area * dhat * kappa / (2 * d * d * dhat) * (d + dhat) * glm::outerProduct(normal, normal);
}

template <typename Scalar>
__forceinline__ __host__ __device__ Scalar trace(const glm::tmat3x3<Scalar>& a)
{
    return a[0][0] + a[1][1] + a[2][2];
}

template <typename Scalar>
__forceinline__ __host__ __device__ Scalar trace2(const glm::tmat3x3<Scalar>& a)
{
    return (a[0][0] * a[0][0]) + (a[1][1] * a[1][1]) + (a[2][2] * a[2][2]);
}

template <typename Scalar>
__forceinline__ __host__ __device__ Scalar trace4(const glm::tmat3x3<Scalar>& a)
{
    return a[0][0] * a[0][0] * a[0][0] * a[0][0] + a[1][1] * a[1][1] * a[1][1] * a[1][1] + a[2][2] * a[2][2] * a[2][2] * a[2][2];
}

template <typename Scalar>
__forceinline__ __host__ __device__ Scalar detDiag(const glm::tmat3x3<Scalar>& a)
{
    return a[0][0] * a[1][1] * a[2][2];
}

template <typename Scalar>
__forceinline__ __host__ __device__ Scalar det2Diag(const glm::tmat3x3<Scalar>& a)
{
    return a[0][0] * a[0][0] * a[1][1] * a[1][1] * a[2][2] * a[2][2];
}

template <typename Scalar>
__forceinline__ __host__ __device__ glm::tmat3x3<Scalar> dI3df(const glm::tmat3x3<Scalar>& F) {
    glm::tvec3<Scalar> f0 = F[0], f1 = F[1], f2 = F[2];
    glm::tmat3x3<Scalar> ret((Scalar)0);
    ret[0] = glm::cross(f1, f2);
    ret[1] = glm::cross(f2, f0);
    ret[2] = glm::cross(f0, f1);
    return ret;
}

__global__ void AddExternal(glm::vec3* V, int numVerts, bool jump, float mass, glm::vec3 vel);

template <typename Scalar>
__forceinline__ __host__ __device__ Scalar frobeniusNorm(const glm::tmat3x3<Scalar>& m) {
    return m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2] +
        m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2] +
        m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2];
}

template <typename Scalar>
__host__ __device__ glm::tmat3x3<Scalar> Build_Edge_Matrix(const glm::tvec3<Scalar>* X, const indexType* Tet, int tet) {
    glm::tmat3x3<Scalar> ret((Scalar)0);
    ret[0] = X[Tet[tet * 4 + 1]] - X[Tet[tet * 4]];
    ret[1] = X[Tet[tet * 4 + 2]] - X[Tet[tet * 4]];
    ret[2] = X[Tet[tet * 4 + 3]] - X[Tet[tet * 4]];

    return ret;
}

template <typename Scalar>
__global__ void computeInvDmV0(Scalar* V0, glm::tmat3x3<Scalar>* DmInv, int numTets, const glm::tvec3<Scalar>* X, const indexType* Tet, Scalar* contact_area, Scalar* degree)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numTets)
    {
        glm::tmat3x3<Scalar> Dm = Build_Edge_Matrix(X, Tet, index);
        DmInv[index] = glm::inverse(Dm);
        V0[index] = glm::abs(glm::determinant(Dm)) / 6.0f;
        const indexType i0 = Tet[index * 4];
        const indexType i1 = Tet[index * 4 + 1];
        const indexType i2 = Tet[index * 4 + 2];
        const indexType i3 = Tet[index * 4 + 3];
        const glm::tvec3<Scalar> x0 = X[i0];
        const glm::tvec3<Scalar> x1 = X[i1];
        const glm::tvec3<Scalar> x2 = X[i2];
        const glm::tvec3<Scalar> x3 = X[i3];
        Scalar area0 = 0.5f * glm::length(glm::cross(x1 - x0, x2 - x0));
        Scalar area1 = 0.5f * glm::length(glm::cross(x2 - x1, x3 - x1));
        Scalar area2 = 0.5f * glm::length(glm::cross(x3 - x2, x0 - x2));
        Scalar area3 = 0.5f * glm::length(glm::cross(x0 - x3, x1 - x3));
        atomicAdd(&contact_area[i0], (area0 + area2 + area3) / 3.0f);
        atomicAdd(&contact_area[i1], (area0 + area1 + area3) / 3.0f);
        atomicAdd(&contact_area[i2], (area0 + area1 + area2) / 3.0f);
        atomicAdd(&contact_area[i3], (area1 + area2 + area3) / 3.0f);
        atomicAdd(&degree[i0], (Scalar)1);
        atomicAdd(&degree[i1], (Scalar)1);
        atomicAdd(&degree[i2], (Scalar)1);
        atomicAdd(&degree[i3], (Scalar)1);
    }
}

template <typename Scalar>
__global__ void computeInvDm(glm::mat3* DmInv, int numTets, const glm::tvec3<Scalar>* X, const indexType* Tet)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numTets)
    {
        glm::mat3 Dm = Build_Edge_Matrix(X, Tet, index);
        DmInv[index] = glm::inverse(Dm);
    }
}

template <typename Scalar>
__global__ void IPCCDKernel(glm::tvec3<Scalar>* X, glm::tvec3<Scalar>* XTilde, glm::tvec3<Scalar>* V, Scalar* tI, glm::vec3* normals, float muT, float muN, int numVerts);

template <typename Scalar>
__global__ void CCDKernel(glm::tvec3<Scalar>* X, glm::tvec3<Scalar>* XTilde, glm::tvec3<Scalar>* V, Scalar* tI, glm::vec3* normals, float muT, float muN, int numVerts, Scalar dt);