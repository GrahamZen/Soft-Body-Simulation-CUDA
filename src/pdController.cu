#include <cuda.h>

#include <sceneStructs.h>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/matrix_transform.hpp> 
#include <simulationContext.h>
#include <utilities.h>
#include <utilities.cuh>
#include <iostream>
#include <deformable_mesh.h>
#include <solver.h>

#include <svd3_cuda.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

/*
__global__ void initAMatrix(int* idx, int* row, int* col, int rowLen, int totalNumber)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < totalNumber)
    {
        row[index] = idx[index] / rowLen;
        col[index] = idx[index] % rowLen;
    }
}



__global__ void computeGlobalIndex(int* row, int* col, int* rc, int totSize, int vertNumber)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < vertNumber)
    {
        int offset = index * 3;
        b[offset + 0] = b[offset + 0] + sn[offset + 0] * massDt_2;
        b[offset + 1] = b[offset + 1] + sn[offset + 1] * massDt_2;
        b[offset + 2] = b[offset + 2] + sn[offset + 2] * massDt_2;
    }
}*/

/*
SoftBody::SoftBody(const char* nodeFileName, const char* eleFileName, SimulationCUDAContext* context, const glm::vec3& pos, const glm::vec3& scale,
    const glm::vec3& rot, float mass, float stiffness_0, float stiffness_1, float damp, float muN, float muT, int constraints, bool centralize, int startIndex)
    : mpSimContext(context), mass(mass), stiffness_0(stiffness_0), stiffness_1(stiffness_1), damp(damp), muN(muN), muT(muT), numConstraints(constraints)
{
    loadNodeFile(nodeFileName, centralize);

    // transform
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, pos);
    model = glm::scale(model, scale);
    model = glm::rotate(model, glm::radians(rot.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
    int threadsPerBlock = 64;
    int blocks = (number + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void**)&X0, sizeof(glm::vec3) * number);
    cudaMemcpy(X0, X, sizeof(glm::vec3) * number, cudaMemcpyDeviceToDevice);

    loadEleFile(eleFileName, startIndex);

    Mesh::tet_number = tet_number;

    InitModel();

    cudaMalloc((void**)&Force, sizeof(glm::vec3) * number);
    cudaMemset(Force, 0, sizeof(glm::vec3) * number);
    cudaMalloc((void**)&V, sizeof(glm::vec3) * number);
    cudaMemset(V, 0, sizeof(glm::vec3) * number);
    cudaMalloc((void**)&inv_Dm, sizeof(glm::mat4) * tet_number);
    cudaMalloc((void**)&V_sum, sizeof(glm::vec3) * number);
    cudaMemset(V_sum, 0, sizeof(glm::vec3) * number);
    createTetrahedron();
    cudaMalloc((void**)&V_num, sizeof(int) * number);
    cudaMemset(V_num, 0, sizeof(int) * number);
    blocks = (tet_number + threadsPerBlock - 1) / threadsPerBlock;
    computeInvDm << < blocks, threadsPerBlock >> > (inv_Dm, tet_number, X, Tet);
}*/
/**
// Should compute SiTAiTAiSi, which is a sparse matrix
// Ai here is I
// size of row, col, val are 48 * tetNumber + vertNumber
// row, col, val are used to initialize sparse matrix SiTSi
__global__ void computeSiTSi(int* outIdx, float* val, glm::mat3* DmInv, int* tetIndex, float weight, int tetNumber, int vertNumber)
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
        glm::mat3 Dm_1 = DmInv[index];

        glm::mat4x3 DmP;
        DmP[0] = Dm_1[0];
        DmP[1] = Dm_1[1];
        DmP[2] = Dm_1[2];
        glm::vec3 ptt = glm::vec3(-Dm_1[0][0] - Dm_1[0][1] - Dm_1[0][2], -Dm_1[1][0] - Dm_1[1][1] - Dm_1[1][2], -Dm_1[2][0] - Dm_1[2][1] - Dm_1[2][2]);
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
                    wrapRowColVal(start + (i * 12 + j * 3 + k), outIdx, val, vc + k, vr + k, st[j][i] * weight, rowLen);
                }
            }
        }
    }
}*/
/*
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

__global__ void addM_h2Sn(float* b, float* sn, float massDt_2, int vertNumber)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < vertNumber)
    {
        int offset = index * 3;
        b[offset + 0] = b[offset + 0] + sn[offset + 0] * massDt_2;
        b[offset + 1] = b[offset + 1] + sn[offset + 1] * massDt_2;
        b[offset + 2] = b[offset + 2] + sn[offset + 2] * massDt_2;
    }
}

*/
/*
// dt2_m_1 is dt^2 / mass
// s(n) = q(n) + dt*v(n) + dt^2 * M^(-1) * fext(n)
__global__ void computeSn(float* sn, float dt, float dt2_m_1, glm::vec3* pos, glm::vec3* vel, glm::vec3* force, int mass_1, int numVerts)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numVerts)
    {
        // sn is of size 3*number
        sn[index * 3 + 0] = pos[index].x + dt * vel[index].x + dt2_m_1 * force[index].x * mass_1;
        sn[index * 3 + 1] = pos[index].y + dt * vel[index].y + dt2_m_1 * force[index].y * mass_1;
        sn[index * 3 + 2] = pos[index].z + dt * vel[index].z + dt2_m_1 * force[index].z * mass_1;
    }
}*/
/*
__global__ void updateVelPos(float* newPos, float dt_1, glm::vec3* pos, glm::vec3* vel, int numVerts)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numVerts)
    {
        int offset = 3 * index;
        glm::vec3 newPosition = glm::vec3(newPos[offset], newPos[offset + 1], newPos[offset + 2]);
        // sn is of size 3*number
        vel[index] = (newPosition - pos[index]) * dt_1;
        pos[index] = newPosition;
    }
}

*/
/*
__global__ void computeLocal(float* V0, float wi, float* xProj, glm::mat3* DmInv, float* qn__1, GLuint* tetIndex, int tetNumber)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < tetNumber)
    {
        //int posStart = index * 12;
        int v0Ind = tetIndex[index * 4 + 0] * 3;
        int v1Ind = tetIndex[index * 4 + 1] * 3;
        int v2Ind = tetIndex[index * 4 + 2] * 3;
        int v3Ind = tetIndex[index * 4 + 3] * 3;
        glm::vec3 v0 = glm::vec3(qn__1[v0Ind + 0], qn__1[v0Ind + 1], qn__1[v0Ind + 2]);
        glm::vec3 v1 = glm::vec3(qn__1[v1Ind + 0], qn__1[v1Ind + 1], qn__1[v1Ind + 2]);
        glm::vec3 v2 = glm::vec3(qn__1[v2Ind + 0], qn__1[v2Ind + 1], qn__1[v2Ind + 2]);
        glm::vec3 v3 = glm::vec3(qn__1[v3Ind + 0], qn__1[v3Ind + 1], qn__1[v3Ind + 2]);

        float weight = V0[index] * wi;

        glm::mat3 Dl = glm::mat3();
        glm::mat3 Dm_1 = DmInv[index];

        Dl[0] = v0 - v3;
        Dl[1] = v1 - v3;
        Dl[2] = v2 - v3;

        glm::mat3 A = Dl * Dm_1;
        glm::mat3 U, S, V;

        //glmSVD(A, U, S, V);

        svd(A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1], A[0][2], A[1][2], A[2][2],
            U[0][0], U[1][0], U[2][0], U[0][1], U[1][1], U[2][1], U[0][2], U[1][2], U[2][2],
            S[0][0], S[1][0], S[2][0], S[0][1], S[1][1], S[2][1], S[0][2], S[1][2], S[2][2],
            V[0][0], V[1][0], V[2][0], V[0][1], V[1][1], V[2][1], V[0][2], V[1][2], V[2][2]);

        glm::mat3 R = U * glm::transpose(V);
        if (glm::determinant(R) < 0)
        {
            R[2] = -R[2];
        }

        glm::mat4x3 DmP;
        DmP[0] = Dm_1[0];
        DmP[1] = Dm_1[1];
        DmP[2] = Dm_1[2];
        glm::vec3 ptt = glm::vec3(-Dm_1[0][0] - Dm_1[1][0] - Dm_1[2][0], -Dm_1[0][1] - Dm_1[1][1] - Dm_1[2][1], -Dm_1[0][2] - Dm_1[1][2] - Dm_1[2][2]);
        DmP[3] = ptt;

        glm::mat3x3 rrr = glm::transpose(R);
        glm::mat4x3 Dm_1r = rrr * DmP;

        for (int i = 0; i < 4; i++)
        {
            int vr = tetIndex[index * 4 + i] * 3;
            atomicAdd(&(xProj[vr + 0]), weight * Dm_1r[i][0]);
            atomicAdd(&(xProj[vr + 1]), weight * Dm_1r[i][1]);
            atomicAdd(&(xProj[vr + 2]), weight * Dm_1r[i][2]);
        }

        /*
        float const& p1 = R[0][0];
        float const& p2 = R[0][1];
        float const& p3 = R[0][2];
        float const& p4 = R[1][0];
        float const& p5 = R[1][1];
        float const& p6 = R[1][2];
        float const& p7 = R[2][0];
        float const& p8 = R[2][1];
        float const& p9 = R[2][2];

        float const& d11 = Dm_1[0][0];
        float const& d21 = Dm_1[0][1];
        float const& d31 = Dm_1[0][2];
        float const& d12 = Dm_1[1][0];
        float const& d22 = Dm_1[1][1];
        float const& d32 = Dm_1[1][2];
        float const& d13 = Dm_1[2][0];
        float const& d23 = Dm_1[2][1];
        float const& d33 = Dm_1[2][2];

        float const _d11_d21_d31 = -d11 - d21 - d31;
        float const _d12_d22_d32 = -d12 - d22 - d32;
        float const _d13_d23_d33 = -d13 - d23 - d33;

        // we have already symbolically computed wi * (Ai*Si)^T * Bi * pi
        float const bi0 = (d11 * p1) + (d12 * p4) + (d13 * p7);
        float const bi1 = (d11 * p2) + (d12 * p5) + (d13 * p8);
        float const bi2 = (d11 * p3) + (d12 * p6) + (d13 * p9);
        float const bj0 = (d21 * p1) + (d22 * p4) + (d23 * p7);
        float const bj1 = (d21 * p2) + (d22 * p5) + (d23 * p8);
        float const bj2 = (d21 * p3) + (d22 * p6) + (d23 * p9);
        float const bk0 = (d31 * p1) + (d32 * p4) + (d33 * p7);
        float const bk1 = (d31 * p2) + (d32 * p5) + (d33 * p8);
        float const bk2 = (d31 * p3) + (d32 * p6) + (d33 * p9);
        float const bl0 = p1 * (_d11_d21_d31)+p4 * (_d12_d22_d32)+p7 * (_d13_d23_d33);
        float const bl1 = p2 * (_d11_d21_d31)+p5 * (_d12_d22_d32)+p8 * (_d13_d23_d33);
        float const bl2 = p3 * (_d11_d21_d31)+p6 * (_d12_d22_d32)+p9 * (_d13_d23_d33);

        atomicAdd(&(xProj[v0Ind + 0]), weight * bi0);
        atomicAdd(&(xProj[v0Ind + 1]), weight * bi1);
        atomicAdd(&(xProj[v0Ind + 2]), weight * bi2);
        atomicAdd(&(xProj[v1Ind + 0]), weight * bj0);
        atomicAdd(&(xProj[v1Ind + 1]), weight * bj1);
        atomicAdd(&(xProj[v1Ind + 2]), weight * bj2);
        atomicAdd(&(xProj[v2Ind + 0]), weight * bk0);
        atomicAdd(&(xProj[v2Ind + 1]), weight * bk1);
        atomicAdd(&(xProj[v2Ind + 2]), weight * bk2);
        atomicAdd(&(xProj[v3Ind + 0]), weight * bl0);
        atomicAdd(&(xProj[v3Ind + 1]), weight * bl1);
        atomicAdd(&(xProj[v3Ind + 2]), weight * bl2);
    }
}
/*
//simple example of gravity
__global__ void setExtForce(glm::vec3* ExtForce, float mass, glm::vec3 gravity, int numVerts)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numVerts)
    {
        ExtForce[index] = gravity * mass;
    }
}*/
/**
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

        glm::mat3 Dl = glm::mat3();

        Dl[0] = v0 - v3;
        Dl[1] = v1 - v3;
        Dl[2] = v2 - v3;
        Dl = glm::transpose(Dl);

        glm::mat3 Dm_1 = glm::transpose(DmInv[index]);

        glm::mat3 A = Dl * Dm_1;

        //glm::mat3 A = Dl * glm::transpose(Dm_1);
        //glm::mat3 A = glm::transpose(Dl) * glm::transpose(Dm_1);
        //glm::mat3 A = Dl * Dm_1;
        //A = glm::transpose(A);
        glm::mat3x3 U;
        glm::mat3x3 V;
        glm::mat3x3 S;

        //glmSVD(A, U, S, V);

        svd(A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1], A[0][2], A[1][2], A[2][2],
            U[0][0], U[1][0], U[2][0], U[0][1], U[1][1], U[2][1], U[0][2], U[1][2], U[2][2],
            S[0][0], S[1][0], S[2][0], S[0][1], S[1][1], S[2][1], S[0][2], S[1][2], S[2][2],
            V[0][0], V[1][0], V[2][0], V[0][1], V[1][1], V[2][1], V[0][2], V[1][2], V[2][2]);

        glm::mat3 R = U * glm::transpose(V);
        if (glm::determinant(R) < 0)
        {
            R[2] = -R[2];
        }

        glm::mat4x3 DmP;
        DmP[0] = Dm_1[0];
        DmP[1] = Dm_1[1];
        DmP[2] = Dm_1[2];
        glm::vec3 ptt = glm::vec3(-Dm_1[0][0] - Dm_1[1][0] - Dm_1[2][0], -Dm_1[0][1] - Dm_1[1][1] - Dm_1[2][1], -Dm_1[0][2] - Dm_1[1][2] - Dm_1[2][2]);
        DmP[3] = ptt;

        glm::mat3x3 rrr = glm::transpose(R);
        glm::mat4x3 Dm_1r = rrr * DmP;

        for (int i = 0; i < 4; i++)
        {
            int vr = tetIndex[index * 4 + i] * 3;
            atomicAdd(&(xProj[vr + 0]), weight * Dm_1r[i][0]);
            atomicAdd(&(xProj[vr + 1]), weight * Dm_1r[i][1]);
            atomicAdd(&(xProj[vr + 2]), weight * Dm_1r[i][2]);
        }
    }
}*/