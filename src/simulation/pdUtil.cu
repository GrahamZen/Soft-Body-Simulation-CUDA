#include <utilities.cuh>
#include <svd3_cuda.h>
#include <cuda.h>

__device__ void svdGLM(const glm::mat3& A, glm::mat3& U, glm::mat3& S, glm::mat3& V)
{
    svd(A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1], A[0][2], A[1][2], A[2][2],
        U[0][0], U[1][0], U[2][0], U[0][1], U[1][1], U[2][1], U[0][2], U[1][2], U[2][2],
        S[0][0], S[1][0], S[2][0], S[0][1], S[1][1], S[2][1], S[0][2], S[1][2], S[2][2],
        V[0][0], V[1][0], V[2][0], V[0][1], V[1][1], V[2][1], V[0][2], V[1][2], V[2][2]);
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
