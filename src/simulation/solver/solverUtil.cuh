#pragma once

#include <matrix.h>
#include <svd3_cuda.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

template<typename Scalar, size_t Rows, size_t Cols>
__inline__ __host__ __device__ void printMatrix(const Matrix<Scalar, Rows, Cols>& m, const char* name) {
    printf("Matrix %d x %d %s\n%f %f %f\n%f %f %f\n%f %f %f\n--------------------------------\n", Rows, Cols, name,
        m[0][0], m[0][1], m[0][2],
        m[1][0], m[1][1], m[1][2],
        m[2][0], m[2][1], m[2][2]);
}

template<typename Scalar, size_t N>
__inline__ __host__ __device__ void printVector(const Vector<Scalar, N>& v, const char* name) {
    printf("Vector %d %s\n%f %f %f \n--------------------------------\n", N, name, v[0], v[1], v[2]);
}

template<typename Scalar>
__inline__ __host__ __device__ void printGLMMatrix(const glm::tmat3x3<Scalar>& m, const char* name) {
    printf("Matrix 3 x 3 %s\n%f %f %f\n%f %f %f\n%f %f %f\n--------------------------------\n", name,
        m[0][0], m[0][1], m[0][2],
        m[1][0], m[1][1], m[1][2],
        m[2][0], m[2][1], m[2][2]);
}

template<typename Scalar>
__inline__ __host__ __device__ void printGLMVector(const glm::tvec3<Scalar>& v, const char* name) {
    printf("Vector 3 %s\n%f %f %f \n--------------------------------\n", name, v.x, v.y, v.z);
}

template <typename Scalar>
__inline__ __host__ __device__ Scalar trace(const glm::tmat3x3<Scalar>& a)
{
    return a[0][0] + a[1][1] + a[2][2];
}

template <typename Scalar>
__inline__ __host__ __device__ Scalar trace2(const glm::tmat3x3<Scalar>& a)
{
    return (a[0][0] * a[0][0]) + (a[1][1] * a[1][1]) + (a[2][2] * a[2][2]);
}

template <typename Scalar>
__inline__ __host__ __device__ Scalar trace4(const glm::tmat3x3<Scalar>& a)
{
    return a[0][0] * a[0][0] * a[0][0] * a[0][0] + a[1][1] * a[1][1] * a[1][1] * a[1][1] + a[2][2] * a[2][2] * a[2][2] * a[2][2];
}

template <typename Scalar>
__inline__ __host__ __device__ Scalar detDiag(const glm::tmat3x3<Scalar>& a)
{
    return a[0][0] * a[1][1] * a[2][2];
}

template <typename Scalar>
__inline__ __host__ __device__ Scalar det2Diag(const glm::tmat3x3<Scalar>& a)
{
    return a[0][0] * a[0][0] * a[1][1] * a[1][1] * a[2][2] * a[2][2];
}

template <typename Scalar>
__inline__ __host__ __device__ glm::tmat3x3<Scalar> dI3df(const glm::tmat3x3<Scalar>& F) {
    glm::tvec3<Scalar> f0 = F[0], f1 = F[1], f2 = F[2];
    glm::tmat3x3<Scalar> ret((Scalar)0);
    ret[0] = glm::cross(f1, f2);
    ret[1] = glm::cross(f2, f0);
    ret[2] = glm::cross(f0, f1);
    return ret;
}


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

__global__ void AddExternal(glm::vec3* V, int numVerts, bool jump, float mass, glm::vec3 vel);

template <typename Scalar>
__inline__ __host__ __device__ Scalar frobeniusNorm(const glm::tmat3x3<Scalar>& m) {
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
__device__ Matrix9x12<Scalar> ComputePFPx(const glm::tmat3x3<Scalar>& DmInv)
{
    const Scalar m = DmInv[0][0];
    const Scalar n = DmInv[1][0];
    const Scalar o = DmInv[2][0];
    const Scalar p = DmInv[0][1];
    const Scalar q = DmInv[1][1];
    const Scalar r = DmInv[2][1];
    const Scalar s = DmInv[0][2];
    const Scalar t = DmInv[1][2];
    const Scalar u = DmInv[2][2];
    const Scalar t1 = -m - p - s;
    const Scalar t2 = -n - q - t;
    const Scalar t3 = -o - r - u;
    Matrix9x12<Scalar> PFPx;
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
__global__ void CCDKernel(glm::tvec3<Scalar>* X, glm::tvec3<Scalar>* XTilde, glm::tvec3<Scalar>* V, Scalar* tI, glm::vec3* normals, float muT, float muN, int numVerts) {
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
        // V[idx] = (Scalar)-muN * velNormal;
        // V[idx] = X[idx] - XTilde[idx];
    }
    else
    {
        X[idx] = XTilde[idx];
    }
    //XTilde[idx] = X[idx];
}