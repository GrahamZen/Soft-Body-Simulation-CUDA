#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <svd3_cuda.h>

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

__inline__ __device__ void svdGLM(const glm::mat3& A, glm::mat3& U, glm::mat3& S, glm::mat3& V)
{
    svd(A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1], A[0][2], A[1][2], A[2][2],
        U[0][0], U[1][0], U[2][0], U[0][1], U[1][1], U[2][1], U[0][2], U[1][2], U[2][2],
        S[0][0], S[1][0], S[2][0], S[0][1], S[1][1], S[2][1], S[0][2], S[1][2], S[2][2],
        V[0][0], V[1][0], V[2][0], V[0][1], V[1][1], V[2][1], V[0][2], V[1][2], V[2][2]);
}

__global__ void AddExternal(glm::vec3* V, int numVerts, bool jump, float mass, glm::vec3 vel);