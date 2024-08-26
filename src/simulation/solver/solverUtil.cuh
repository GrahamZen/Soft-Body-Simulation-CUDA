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

__inline__ __device__ void svdGLM(const glm::mat3& A, glm::mat3& U, glm::mat3& S, glm::mat3& V)
{
    svd(A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1], A[0][2], A[1][2], A[2][2],
        U[0][0], U[1][0], U[2][0], U[0][1], U[1][1], U[2][1], U[0][2], U[1][2], U[2][2],
        S[0][0], S[1][1], S[2][2],
        V[0][0], V[1][0], V[2][0], V[0][1], V[1][1], V[2][1], V[0][2], V[1][2], V[2][2]);
}

__global__ void AddExternal(glm::vec3* V, int numVerts, bool jump, float mass, glm::vec3 vel);

template <typename HighP>
__device__ glm::mat3 Build_Edge_Matrix(const glm::tvec3<HighP>* X, const indexType* Tet, int tet) {
    glm::mat3 ret(0.0f);
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