#pragma once

#include <svd3_cuda.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

__inline__ __device__ float trace(const glm::mat3& a)
{
    return a[0][0] + a[1][1] + a[2][2];
}

__inline__ __device__ float trace2(const glm::mat3& a)
{
    return (float)((a[0][0] * a[0][0]) + (a[1][1] * a[1][1]) + (a[2][2] * a[2][2]));
}

__inline__ __device__ float trace4(const glm::mat3& a)
{
    return (float)(a[0][0] * a[0][0] * a[0][0] * a[0][0] + a[1][1] * a[1][1] * a[1][1] * a[1][1] + a[2][2] * a[2][2] * a[2][2] * a[2][2]);
}

__inline__ __device__ float det2(const glm::mat3& a)
{
    return (float)(a[0][0] * a[0][0] * a[1][1] * a[1][1] * a[2][2] * a[2][2]);
}

template <typename HighP>
__inline__ __device__ void svdGLM(const glm::tmat3x3<HighP>& A, glm::tmat3x3<HighP>& U, glm::tmat3x3<HighP>& S, glm::tmat3x3<HighP>& V)
{
    svd(A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1], A[0][2], A[1][2], A[2][2],
        U[0][0], U[1][0], U[2][0], U[0][1], U[1][1], U[2][1], U[0][2], U[1][2], U[2][2],
        S[0][0], S[1][1], S[2][2],
        V[0][0], V[1][0], V[2][0], V[0][1], V[1][1], V[2][1], V[0][2], V[1][2], V[2][2]);
}

__global__ void AddExternal(glm::vec3* V, int numVerts, bool jump, float mass, glm::vec3 vel);

template <typename HighP>
__inline__ __device__ HighP frobeniusNorm(const glm::tmat3x3<HighP>& m) {
    return sqrt(m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2] +
        m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2] +
        m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2]);
}

template <typename HighP>
__device__ glm::tmat3x3<HighP> Build_Edge_Matrix(const glm::tvec3<HighP>* X, const indexType* Tet, int tet) {
    glm::tmat3x3<HighP> ret((HighP)0);
    ret[0] = X[Tet[tet * 4]] - X[Tet[tet * 4 + 3]];
    ret[1] = X[Tet[tet * 4 + 1]] - X[Tet[tet * 4 + 3]];
    ret[2] = X[Tet[tet * 4 + 2]] - X[Tet[tet * 4 + 3]];

    return ret;
}

template <typename HighP>
__global__ void computeInvDmV0(float* V0, glm::mat3* inv_Dm, int numTets, const glm::tvec3<HighP>* X, const indexType* Tet)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numTets)
    {
        glm::mat3 Dm = Build_Edge_Matrix(X, Tet, index);
        inv_Dm[index] = glm::inverse(Dm);
        V0[index] = glm::abs(glm::determinant(Dm)) / 6.0f;
    }
}

template <typename HighP>
__global__ void computeInvDm(glm::mat3* inv_Dm, int numTets, const glm::tvec3<HighP>* X, const indexType* Tet)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numTets)
    {
        glm::mat3 Dm = Build_Edge_Matrix(X, Tet, index);
        inv_Dm[index] = glm::inverse(Dm);
    }
}