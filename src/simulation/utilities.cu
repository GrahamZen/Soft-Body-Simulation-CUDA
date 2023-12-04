#include <utilities.cuh>
#include <svd3_cuda.h>
#include <cuda.h>
#include <bvh.h>

#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

void inspectMortonCodes(int* dev_mortonCodes, int numTets) {
    std::vector<unsigned int> hstMorton(numTets);
    cudaMemcpy(hstMorton.data(), dev_mortonCodes, numTets * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    inspectHostMorton(hstMorton.data(), numTets);
}

void inspectBVHNode(BVHNode* dev_BVHNodes, int numTets)
{
    std::vector<BVHNode> hstBVHNodes(2 * numTets - 1);
    cudaMemcpy(hstBVHNodes.data(), dev_BVHNodes, sizeof(BVHNode) * (2 * numTets - 1), cudaMemcpyDeviceToHost);
    inspectHost(hstBVHNodes.data(), 2 * numTets - 1);
}

void inspectBVH(AABB* dev_aabbs, int size)
{
    std::vector<AABB> hstAABB(size);
    cudaMemcpy(hstAABB.data(), dev_aabbs, sizeof(AABB) * size, cudaMemcpyDeviceToHost);
    inspectHost(hstAABB.data(), size);
}

void inspectQuerys(Query* dev_query, int size)
{
    std::vector<Query> hstAABB(size);
    cudaMemcpy(hstAABB.data(), dev_query, sizeof(Query) * size, cudaMemcpyDeviceToHost);
    inspectHost(hstAABB.data(), size);
}

__global__ void TransformVertices(glm::vec3* X, glm::mat4 transform, int numVerts)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numVerts)
    {
        X[index] = glm::vec3(transform * glm::vec4(X[index], 1.f));
    }
}

__global__ void AddExternal(glm::vec3* V, int numVerts, bool jump, float mass, glm::vec3 vel)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numVerts)
    {
        if (jump)
            V[index] += vel / mass;
    }
}

__device__ glm::mat3 Build_Edge_Matrix(const glm::vec3* X, const GLuint* Tet, int tet) {
    glm::mat3 ret(0.0f);
    ret[0] = X[Tet[tet * 4]] - X[Tet[tet * 4 + 3]];
    ret[1] = X[Tet[tet * 4 + 1]] - X[Tet[tet * 4 + 3]];
    ret[2] = X[Tet[tet * 4 + 2]] - X[Tet[tet * 4 + 3]];

    return ret;
}

__device__ void svdGLM(const glm::mat3& A, glm::mat3& U, glm::mat3& S, glm::mat3& V)
{
    svd(A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1], A[0][2], A[1][2], A[2][2],
        U[0][0], U[1][0], U[2][0], U[0][1], U[1][1], U[2][1], U[0][2], U[1][2], U[2][2],
        S[0][0], S[1][0], S[2][0], S[0][1], S[1][1], S[2][1], S[0][2], S[1][2], S[2][2],
        V[0][0], V[1][0], V[2][0], V[0][1], V[1][1], V[2][1], V[0][2], V[1][2], V[2][2]);
}

__global__ void computeInvDmV0(float* V0, glm::mat3* inv_Dm, int numTets, const glm::vec3* X, const GLuint* Tet)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numTets)
    {
        glm::mat3 Dm = Build_Edge_Matrix(X, Tet, index);
        inv_Dm[index] = glm::inverse(Dm);
        V0[index] = glm::abs(glm::determinant(Dm)) / 6.0f;
    }
}

__global__ void LaplacianGatherKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int numTets, const GLuint* Tet) {
    int tet = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tet < numTets) {
        glm::vec3 sum = V[Tet[tet * 4]] + V[Tet[tet * 4 + 1]] + V[Tet[tet * 4 + 2]] + V[Tet[tet * 4 + 3]];

        for (int i = 0; i < 4; ++i) {
            int idx = Tet[tet * 4 + i];
            atomicAdd(&(V_sum[idx].x), sum.x - V[idx].x);
            atomicAdd(&(V_sum[idx].y), sum.y - V[idx].y);
            atomicAdd(&(V_sum[idx].z), sum.z - V[idx].z);
            atomicAdd(&(V_num[idx]), 3);
        }
    }
}

__global__ void LaplacianKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int numVerts, const GLuint* Tet, float blendAlpha) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < numVerts) {
        V[i] = blendAlpha * V[i] + (1 - blendAlpha) * V_sum[i] / float(V_num[i]);
    }
}


__global__ void PopulatePos(glm::vec3* vertices, glm::vec3* X, GLuint* Tet, int numTets)
{
    int tet = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tet < numTets)
    {
        vertices[tet * 12 + 0] = X[Tet[tet * 4 + 0]];
        vertices[tet * 12 + 1] = X[Tet[tet * 4 + 2]];
        vertices[tet * 12 + 2] = X[Tet[tet * 4 + 1]];
        vertices[tet * 12 + 3] = X[Tet[tet * 4 + 0]];
        vertices[tet * 12 + 4] = X[Tet[tet * 4 + 3]];
        vertices[tet * 12 + 5] = X[Tet[tet * 4 + 2]];
        vertices[tet * 12 + 6] = X[Tet[tet * 4 + 0]];
        vertices[tet * 12 + 7] = X[Tet[tet * 4 + 1]];
        vertices[tet * 12 + 8] = X[Tet[tet * 4 + 3]];
        vertices[tet * 12 + 9] = X[Tet[tet * 4 + 1]];
        vertices[tet * 12 + 10] = X[Tet[tet * 4 + 2]];
        vertices[tet * 12 + 11] = X[Tet[tet * 4 + 3]];
    }
}

__global__ void PopulateTriPos(glm::vec3* vertices, glm::vec3* X, GLuint* Tet, int numTris)
{
    int tri = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tri < numTris)
    {
        vertices[tri * 3 + 0] = X[Tet[tri * 3 + 0]];
        vertices[tri * 3 + 1] = X[Tet[tri * 3 + 2]];
        vertices[tri * 3 + 2] = X[Tet[tri * 3 + 1]];
    }
}

__global__ void RecalculateNormals(glm::vec4* norms, glm::vec3* vertices, int numVerts)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numVerts)
    {
        glm::vec3 v0v1 = vertices[index * 3 + 1] - vertices[index * 3 + 0];
        glm::vec3 v0v2 = vertices[index * 3 + 2] - vertices[index * 3 + 0];
        glm::vec3 nor = glm::cross(v0v1, v0v2);
        norms[index * 3 + 0] = glm::vec4(glm::normalize(nor), 0.f);
        norms[index * 3 + 1] = glm::vec4(glm::normalize(nor), 0.f);
        norms[index * 3 + 2] = glm::vec4(glm::normalize(nor), 0.f);
    }
}

__global__ void ComputeForces(glm::vec3* Force, const glm::vec3* X, const GLuint* Tet, int numTets, const glm::mat3* inv_Dm, float stiffness_0, float stiffness_1) {
    int tet = blockIdx.x * blockDim.x + threadIdx.x;
    if (tet >= numTets) return;

    glm::mat3 F = Build_Edge_Matrix(X, Tet, tet) * inv_Dm[tet];
    glm::mat3 FtF = glm::transpose(F) * F;
    glm::mat3 G = (FtF - glm::mat3(1.0f)) * 0.5f;
    glm::mat3 S = G * (2.0f * stiffness_1) + glm::mat3(1.0f) * (stiffness_0 * trace(G));
    glm::mat3 forces = F * S * glm::transpose(inv_Dm[tet]) * (-1.0f / (6.0f * glm::determinant(inv_Dm[tet])));

    glm::vec3 force_0 = -glm::vec3(forces[0] + forces[1] + forces[2]);
    glm::vec3 force_1 = glm::vec3(forces[0]);
    glm::vec3 force_2 = glm::vec3(forces[1]);
    glm::vec3 force_3 = glm::vec3(forces[2]);

    atomicAdd(&(Force[Tet[tet * 4 + 0]].x), force_0.x);
    atomicAdd(&(Force[Tet[tet * 4 + 0]].y), force_0.y);
    atomicAdd(&(Force[Tet[tet * 4 + 0]].z), force_0.z);
    atomicAdd(&(Force[Tet[tet * 4 + 1]].x), force_1.x);
    atomicAdd(&(Force[Tet[tet * 4 + 1]].y), force_1.y);
    atomicAdd(&(Force[Tet[tet * 4 + 1]].z), force_1.z);
    atomicAdd(&(Force[Tet[tet * 4 + 2]].x), force_2.x);
    atomicAdd(&(Force[Tet[tet * 4 + 2]].y), force_2.y);
    atomicAdd(&(Force[Tet[tet * 4 + 2]].z), force_2.z);
    atomicAdd(&(Force[Tet[tet * 4 + 3]].x), force_3.x);
    atomicAdd(&(Force[Tet[tet * 4 + 3]].y), force_3.y);
    atomicAdd(&(Force[Tet[tet * 4 + 3]].z), force_3.z);
}

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

__global__ void HandleFloorCollision(glm::vec3* X, glm::vec3* V,
    int numVerts, glm::vec3 floorPos, glm::vec3 floorUp, float muT, float muN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVerts) return;
    V[i] *= 0.99f;
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


// Should compute SiTAiTAiSi, which is a sparse matrix
// Ai here is I
// size of row, col, val are 48 * tetNumber + vertNumber
// row, col, val are used to initialize sparse matrix SiTSi
__global__ void computeSiTSi(int* outIdx, float* val, float* V0, glm::mat3* DmInv, GLuint* tetIndex, float weight, int tetNumber, int vertNumber)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < tetNumber)
    {
        //int posStart = index * 12;
        int v0Ind = tetIndex[index * 4 + 0] * 3;
        int v1Ind = tetIndex[index * 4 + 1] * 3;
        int v2Ind = tetIndex[index * 4 + 2] * 3;
        int v3Ind = tetIndex[index * 4 + 3] * 3;

        // there are tetNumber of Dm_1 in total
        glm::mat3 Dm_1 = glm::transpose(DmInv[index]);

        glm::mat4x3 DmP;
        DmP[0] = Dm_1[0];
        DmP[1] = Dm_1[1];
        DmP[2] = Dm_1[2];
        glm::vec3 ptt = glm::vec3(-Dm_1[0][0] - Dm_1[1][0] - Dm_1[2][0], -Dm_1[0][1] - Dm_1[1][1] - Dm_1[2][1], -Dm_1[0][2] - Dm_1[1][2] - Dm_1[2][2]);
        DmP[3] = ptt;
        glm::mat3x4 DmPT = glm::transpose(DmP);
        glm::mat4x4 st = DmPT * DmP;

        int start = index * 48;
        int rowLen = 3 * vertNumber;
        for (int i = 0; i < 4; i++)
        {
            int vr = tetIndex[index * 4 + i] * 3;
            for (int j = 0; j < 4; j++)
            {
                int vc = tetIndex[index * 4 + j] * 3;
                for (int k = 0; k < 3; k++)
                {
                    wrapRowColVal(start + (i * 12 + j * 3 + k), outIdx, val, vc + k, vr + k, st[j][i] * weight * V0[index], rowLen);
                }
            }
        }
    }
}

__global__ void setMDt_2(int* outIdx, float* val, int startIndex, float massDt_2, int vertNumber)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < vertNumber)
    {
        int offset = index * 3;
        int rowLen = vertNumber * 3;
        wrapRowColVal(startIndex + offset + 0, outIdx, val, offset, offset, massDt_2, rowLen);
        wrapRowColVal(startIndex + offset + 1, outIdx, val, offset + 1, offset + 1, massDt_2, rowLen);
        wrapRowColVal(startIndex + offset + 2, outIdx, val, offset + 2, offset + 2, massDt_2, rowLen);
    }
}

__global__ void initAMatrix(int* idx, int* row, int* col, int rowLen, int totalNumber)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < totalNumber)
    {
        row[index] = idx[index] / rowLen;
        col[index] = idx[index] % rowLen;
    }
}

// dt2_m_1 is dt^2 / mass
// s(n) = q(n) + dt*v(n) + dt^2 * M^(-1) * fext(n)
__global__ void computeSn(float* sn, float dt, float dt2_m_1, glm::vec3* pos, glm::vec3* vel, glm::vec3* force, int numVerts)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numVerts)
    {
        // sn is of size 3*numVerts
        sn[index * 3 + 0] = pos[index].x + dt * vel[index].x + dt2_m_1 * force[index].x;
        sn[index * 3 + 1] = pos[index].y + dt * vel[index].y + dt2_m_1 * force[index].y;
        sn[index * 3 + 2] = pos[index].z + dt * vel[index].z + dt2_m_1 * force[index].z;
    }
}

__global__ void computeLocal(float* V0, float wi, float* xProj, glm::mat3* DmInv, float* qn__1, GLuint* tetIndex, int tetNumber)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < tetNumber)
    {
        int v0Ind = tetIndex[index * 4 + 0] * 3;
        int v1Ind = tetIndex[index * 4 + 1] * 3;
        int v2Ind = tetIndex[index * 4 + 2] * 3;
        int v3Ind = tetIndex[index * 4 + 3] * 3;
        glm::vec3 v0 = glm::vec3(qn__1[v0Ind + 0], qn__1[v0Ind + 1], qn__1[v0Ind + 2]);
        glm::vec3 v1 = glm::vec3(qn__1[v1Ind + 0], qn__1[v1Ind + 1], qn__1[v1Ind + 2]);
        glm::vec3 v2 = glm::vec3(qn__1[v2Ind + 0], qn__1[v2Ind + 1], qn__1[v2Ind + 2]);
        glm::vec3 v3 = glm::vec3(qn__1[v3Ind + 0], qn__1[v3Ind + 1], qn__1[v3Ind + 2]);

        float weight = glm::abs(V0[index]) * wi;
        glm::mat3 Dm_1 = DmInv[index];

        glm::mat3 Dl = glm::mat3();
        Dl[0] = v0 - v3;
        Dl[1] = v1 - v3;
        Dl[2] = v2 - v3;

        glm::mat3 A = Dl * Dm_1;

        glm::mat3x3 U1;
        glm::mat3x3 V1;
        glm::mat3x3 S1;
        glm::mat3x3 R1;

        svd(A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1], A[0][2], A[1][2], A[2][2],
            U1[0][0], U1[1][0], U1[2][0], U1[0][1], U1[1][1], U1[2][1], U1[0][2], U1[1][2], U1[2][2],
            S1[0][0], S1[1][0], S1[2][0], S1[0][1], S1[1][1], S1[2][1], S1[0][2], S1[1][2], S1[2][2],
            V1[0][0], V1[1][0], V1[2][0], V1[0][1], V1[1][1], V1[2][1], V1[0][2], V1[1][2], V1[2][2]);

        R1 = U1 * glm::transpose(V1);


        if (glm::determinant(R1) < 0)
        {
            R1[2] = -R1[2];
        }

        glm::mat4x3 DmP;
        DmP[0] = glm::transpose(Dm_1)[0];
        DmP[1] = glm::transpose(Dm_1)[1];
        DmP[2] = glm::transpose(Dm_1)[2];
        glm::vec3 ptt = glm::vec3(-Dm_1[0][0] - Dm_1[0][1] - Dm_1[0][2], -Dm_1[1][0] - Dm_1[1][1] - Dm_1[1][2], -Dm_1[2][0] - Dm_1[2][1] - Dm_1[2][2]);
        DmP[3] = ptt;
        glm::mat4x3 Dm_1r = R1 * DmP;

        for (int i = 0; i < 4; i++)
        {
            int vr = tetIndex[index * 4 + i] * 3;
            atomicAdd(&(xProj[vr + 0]), weight * Dm_1r[i][0]);
            atomicAdd(&(xProj[vr + 1]), weight * Dm_1r[i][1]);
            atomicAdd(&(xProj[vr + 2]), weight * Dm_1r[i][2]);
        }
    }
}

__global__ void computeM_h2Sn(float* b, float* sn, float massDt_2, int vertNumber)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < vertNumber)
    {
        int offset = index * 3;
        b[offset + 0] = sn[offset + 0] * massDt_2;
        b[offset + 1] = sn[offset + 1] * massDt_2;
        b[offset + 2] = sn[offset + 2] * massDt_2;
    }
}

__global__ void addM_h2Sn(float* b, float* masses, int vertNumber)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < vertNumber)
    {
        int offset = index * 3;
        b[offset + 0] = b[offset + 0] + masses[offset + 0];
        b[offset + 1] = b[offset + 1] + masses[offset + 1];
        b[offset + 2] = b[offset + 2] + masses[offset + 2];
    }
}

__global__ void updateVelPos(float* newPos, float dt_1, glm::vec3* pos, glm::vec3* vel, int numVerts)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numVerts)
    {
        int offset = 3 * index;
        glm::vec3 newPosition = glm::vec3(newPos[offset], newPos[offset + 1], newPos[offset + 2]);
        // sn is of size 3*numVerts
        vel[index] = (newPosition - pos[index]) * dt_1;
        pos[index] = newPosition;
    }
}

//simple example of gravity
__global__ void setExtForce(glm::vec3* ExtForce, glm::vec3 gravity, int numVerts)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numVerts)
    {
        ExtForce[index] = gravity;
    }
}

__global__ void CCDKernel(glm::vec3* X, glm::vec3* XTilt, glm::vec3* V, dataType* tI, glm::vec3* normals, float muT, float muN, int numVerts) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numVerts) return;
    float interval = glm::length(XTilt - X);

    if (tI[idx] < 1.0f)
    {
        glm::vec3 normal = normals[idx];
        glm::vec3 vel = XTilt[idx] - X[idx];
        glm::vec3 velNormal = glm::dot(vel, normal) * normal;
        glm::vec3 vT = vel - velNormal;
        float mag_vT = glm::length(vT);
        float a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(velNormal) / mag_vT, 0.0f);
        V[idx] = -muN * velNormal + a * vT;
    }
    else
    {
        X[idx] = XTilt[idx];
    }
}

__global__ void populateBVHNodeAABBPos(BVHNode* nodes, glm::vec3* pos, int numNodes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numNodes) return;
    const AABB& aabb = nodes[idx].bbox;
    pos[idx * 8 + 0] = glm::vec3(aabb.min.x, aabb.min.y, aabb.max.z);
    pos[idx * 8 + 1] = glm::vec3(aabb.max.x, aabb.min.y, aabb.max.z);
    pos[idx * 8 + 2] = glm::vec3(aabb.max.x, aabb.max.y, aabb.max.z);
    pos[idx * 8 + 3] = glm::vec3(aabb.min.x, aabb.max.y, aabb.max.z);
    pos[idx * 8 + 4] = glm::vec3(aabb.min.x, aabb.min.y, aabb.min.z);
    pos[idx * 8 + 5] = glm::vec3(aabb.max.x, aabb.min.y, aabb.min.z);
    pos[idx * 8 + 6] = glm::vec3(aabb.max.x, aabb.max.y, aabb.min.z);
    pos[idx * 8 + 7] = glm::vec3(aabb.min.x, aabb.max.y, aabb.min.z);
}