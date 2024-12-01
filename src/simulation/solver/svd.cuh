#pragma once

#include <svd3_cuda.h>
#include <glm/glm.hpp>


template <typename Scalar>
__inline__ __host__ __device__ void svdGLM(const glm::tmat3x3<Scalar>& A, glm::tmat3x3<Scalar>& U, glm::tmat3x3<Scalar>& S, glm::tmat3x3<Scalar>& V)
{
    svd(A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1], A[0][2], A[1][2], A[2][2],
        U[0][0], U[1][0], U[2][0], U[0][1], U[1][1], U[2][1], U[0][2], U[1][2], U[2][2],
        S[0][0], S[1][1], S[2][2],
        V[0][0], V[1][0], V[2][0], V[0][1], V[1][1], V[2][1], V[0][2], V[1][2], V[2][2]);
}

template <typename Scalar>
__device__ void svdRV(const glm::tmat3x3<Scalar>& A, glm::tmat3x3<Scalar>& U, glm::tmat3x3<Scalar>& S, glm::tmat3x3<Scalar>& V) {
    svdGLM(A, U, S, V);
    glm::tmat3x3<Scalar> L(1);
    L[2][2] = glm::determinant(U * glm::transpose(V));

    Scalar detU = glm::determinant(U);
    Scalar detV = glm::determinant(V);

    if (detU < 0 && detV > 0)
        U = U * L;
    else if (detU > 0 && detV < 0)
        V = V * L;

    S = S * L;
}

template <typename Scalar>
__device__ void polarDecomposition(const glm::tmat3x3<Scalar>& F, glm::tmat3x3<Scalar>& R, glm::tmat3x3<Scalar>& S) {
    glm::tmat3x3<Scalar> U, V;
    svdRV(F, U, S, V);
    R = U * glm::transpose(V);
    S = V * S * glm::transpose(V);
}