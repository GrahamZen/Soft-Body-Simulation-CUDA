#pragma once

#include <matrix.h>
#include <svd3_cuda.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

template<typename HighP, size_t Rows, size_t Cols>
__inline__ __device__ void printMatrix(const Matrix<HighP, Rows, Cols>& m, const char* name) {
    printf("Matrix %d x %d %s\n", Rows, Cols, name);
    for (size_t i = 0; i < Rows; i++) {
        for (size_t j = 0; j < Cols; j++) {
            printf("%f ", m[i][j]);
        }
        printf("\n");
    }
    printf("--------------------------------\n");
}

template<typename HighP, size_t N>
__inline__ __device__ void printVector(const Vector<HighP, N>& v, const char* name) {
    printf("Vector %d %s\n", N, name);
    for (size_t i = 0; i < N; i++) {
        printf("%f ", v[i]);
    }
    printf("\n--------------------------------\n");
}

template<typename HighP>
__inline__ __device__ void printGLMMatrix(const glm::tmat3x3<HighP>& m, const char* name) {
    printf("Matrix 3 x 3 %s\n", name);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            printf("%f ", m[j][i]);
        }
        printf("\n");
    }
    printf("--------------------------------\n");
}

template <typename HighP>
__inline__ __device__ HighP trace(const glm::tmat3x3<HighP>& a)
{
    return a[0][0] + a[1][1] + a[2][2];
}

template <typename HighP>
__inline__ __device__ HighP trace2(const glm::tmat3x3<HighP>& a)
{
    return (a[0][0] * a[0][0]) + (a[1][1] * a[1][1]) + (a[2][2] * a[2][2]);
}

template <typename HighP>
__inline__ __device__ HighP trace4(const glm::tmat3x3<HighP>& a)
{
    return a[0][0] * a[0][0] * a[0][0] * a[0][0] + a[1][1] * a[1][1] * a[1][1] * a[1][1] + a[2][2] * a[2][2] * a[2][2] * a[2][2];
}

template <typename HighP>
__inline__ __device__ HighP detDiag(const glm::tmat3x3<HighP>& a)
{
    return a[0][0] * a[1][1] * a[2][2];
}

template <typename HighP>
__inline__ __device__ HighP det2Diag(const glm::tmat3x3<HighP>& a)
{
    return a[0][0] * a[0][0] * a[1][1] * a[1][1] * a[2][2] * a[2][2];
}

template <typename HighP>
__inline__ __device__ glm::tmat3x3<HighP> dI3df(const glm::tmat3x3<HighP>& F) {
    glm::tvec3<HighP> f0 = F[0], f1 = F[1], f2 = F[2];
    glm::tmat3x3<HighP> ret((HighP)0);
    ret[0] = glm::cross(f1, f2);
    ret[1] = glm::cross(f2, f0);
    ret[2] = glm::cross(f0, f1);
    return ret;
}


template <typename HighP>
__inline__ __device__ void svdGLM(const glm::tmat3x3<HighP>& A, glm::tmat3x3<HighP>& U, glm::tmat3x3<HighP>& S, glm::tmat3x3<HighP>& V)
{
    svd(A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1], A[0][2], A[1][2], A[2][2],
        U[0][0], U[1][0], U[2][0], U[0][1], U[1][1], U[2][1], U[0][2], U[1][2], U[2][2],
        S[0][0], S[1][1], S[2][2],
        V[0][0], V[1][0], V[2][0], V[0][1], V[1][1], V[2][1], V[0][2], V[1][2], V[2][2]);
}

template <typename HighP>
__device__ void svdRV(const glm::tmat3x3<HighP>& A, glm::tmat3x3<HighP>& U, glm::tmat3x3<HighP>& S, glm::tmat3x3<HighP>& V) {
    svdGLM(A, U, S, V);
    glm::tmat3x3<HighP> L(1);
    L[2][2] = glm::determinant(U * glm::transpose(V));

    HighP detU = glm::determinant(U);
    HighP detV = glm::determinant(V);

    if (detU < 0 && detV > 0)
        U = U * L;
    else if (detU > 0 && detV < 0)
        V = V * L;

    S = S * L;
}

template <typename HighP>
__device__ void polarDecomposition(const glm::tmat3x3<HighP>& F, glm::tmat3x3<HighP>& R, glm::tmat3x3<HighP>& S) {
    glm::tmat3x3<HighP> U, V;
    svdRV(F, U, S, V);
    R = U * glm::transpose(V);
    S = V * S * glm::transpose(V);
}

__global__ void AddExternal(glm::vec3* V, int numVerts, bool jump, float mass, glm::vec3 vel);

template <typename HighP>
__inline__ __device__ HighP frobeniusNorm(const glm::tmat3x3<HighP>& m) {
    return m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2] +
        m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2] +
        m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2];
}

template <typename HighP>
__device__ glm::tmat3x3<HighP> Build_Edge_Matrix(const glm::tvec3<HighP>* X, const indexType* Tet, int tet) {
    glm::tmat3x3<HighP> ret((HighP)0);
    ret[0] = X[Tet[tet * 4 + 1]] - X[Tet[tet * 4]];
    ret[1] = X[Tet[tet * 4 + 2]] - X[Tet[tet * 4]];
    ret[2] = X[Tet[tet * 4 + 3]] - X[Tet[tet * 4]];

    return ret;
}

template <typename HighP>
__global__ void computeInvDmV0(HighP* V0, glm::tmat3x3<HighP>* DmInv, int numTets, const glm::tvec3<HighP>* X, const indexType* Tet)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numTets)
    {
        glm::tmat3x3<HighP> Dm = Build_Edge_Matrix(X, Tet, index);
        DmInv[index] = glm::inverse(Dm);
        V0[index] = glm::abs(glm::determinant(Dm)) / 6.0f;
    }
}

template <typename HighP>
__global__ void computeInvDm(glm::mat3* DmInv, int numTets, const glm::tvec3<HighP>* X, const indexType* Tet)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numTets)
    {
        glm::mat3 Dm = Build_Edge_Matrix(X, Tet, index);
        DmInv[index] = glm::inverse(Dm);
    }
}

template <typename HighP>
__device__ Matrix9x12<HighP> ComputePFPx(const glm::tmat3x3<HighP>& DmInv)
{
    const HighP m = DmInv[0][0];
    const HighP n = DmInv[1][0];
    const HighP o = DmInv[2][0];
    const HighP p = DmInv[0][1];
    const HighP q = DmInv[1][1];
    const HighP r = DmInv[2][1];
    const HighP s = DmInv[0][2];
    const HighP t = DmInv[1][2];
    const HighP u = DmInv[2][2];
    const HighP t1 = -m - p - s;
    const HighP t2 = -n - q - t;
    const HighP t3 = -o - r - u;
    Matrix9x12<HighP> PFPx;
    PFPx[0][0] = t1;
    PFPx[0][3] = m;
    PFPx[0][6] = p;
    PFPx[0][9] = s;
    PFPx[1][1] = t1;
    PFPx[1][4] = m;
    PFPx[1][7] = p;
    PFPx[1][10] = s;
    PFPx[2][2] = t1;
    PFPx[2][5] = m;
    PFPx[2][8] = p;
    PFPx[2][11] = s;
    PFPx[3][0] = t2;
    PFPx[3][3] = n;
    PFPx[3][6] = q;
    PFPx[3][9] = t;
    PFPx[4][1] = t2;
    PFPx[4][4] = n;
    PFPx[4][7] = q;
    PFPx[4][10] = t;
    PFPx[5][2] = t2;
    PFPx[5][5] = n;
    PFPx[5][8] = q;
    PFPx[5][11] = t;
    PFPx[6][0] = t3;
    PFPx[6][3] = o;
    PFPx[6][6] = r;
    PFPx[6][9] = u;
    PFPx[7][1] = t3;
    PFPx[7][4] = o;
    PFPx[7][7] = r;
    PFPx[7][10] = u;
    PFPx[8][2] = t3;
    PFPx[8][5] = o;
    PFPx[8][8] = r;
    PFPx[8][11] = u;
    return PFPx;
}

template <typename HighP>
__global__ void IPCCDKernel(glm::tvec3<HighP>* X, glm::tvec3<HighP>* XTilde, glm::tvec3<HighP>* V, colliPrecision* tI, glm::vec3* normals, float muT, float muN, int numVerts) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numVerts) return;
    HighP interval = glm::length(XTilde - X);

    if (tI[idx] < 1.0f)
    {
        glm::tvec3<HighP> normal = normals[idx];
        glm::tvec3<HighP> vel = XTilde[idx] - X[idx];
        glm::tvec3<HighP> velNormal = glm::dot(vel, normal) * normal;
        glm::tvec3<HighP> vT = vel - velNormal;
        HighP mag_vT = glm::length(vT);
        //HighP a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(velNormal) / mag_vT, 0.0);
        V[idx] = (HighP)-muN * velNormal;
        // V[idx] = X[idx] - XTilde[idx];
    }
    else
    {
        X[idx] = XTilde[idx];
    }
    //XTilde[idx] = X[idx];
}

template <typename HighP>
__global__ void CCDKernel(glm::tvec3<HighP>* X, glm::tvec3<HighP>* XTilde, glm::tvec3<HighP>* V, colliPrecision* tI, glm::vec3* normals, float muT, float muN, int numVerts) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numVerts) return;
    HighP interval = glm::length(XTilde - X);

    if (tI[idx] < 1.0f)
    {
        glm::tvec3<HighP> normal = normals[idx];
        glm::tvec3<HighP> vel = XTilde[idx] - X[idx];
        glm::tvec3<HighP> velNormal = glm::dot(vel, normal) * normal;
        glm::tvec3<HighP> vT = vel - velNormal;
        HighP mag_vT = glm::length(vT);
        //HighP a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(velNormal) / mag_vT, 0.0);
        // V[idx] = (HighP)-muN * velNormal;
        // V[idx] = X[idx] - XTilde[idx];
    }
    else
    {
        X[idx] = XTilde[idx];
    }
    //XTilde[idx] = X[idx];
}