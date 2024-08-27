#include <def.h>
#include <simulation/solver/projective/pdUtil.cuh>

namespace PdUtil {
    __global__ void CCDKernel(glm::vec3* X, glm::vec3* XTilde, glm::vec3* V, colliPrecision* tI, glm::vec3* normals, float muT, float muN, int numVerts) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= numVerts) return;
        float interval = glm::length(XTilde - X);

        if (tI[idx] < 1.0f)
        {
            glm::vec3 normal = normals[idx];
            glm::vec3 vel = XTilde[idx] - X[idx];
            glm::vec3 velNormal = glm::dot(vel, normal) * normal;
            glm::vec3 vT = vel - velNormal;
            float mag_vT = glm::length(vT);
            float a = mag_vT == 0 ? 0 : glm::max(1 - muT * (1 + muN) * glm::length(velNormal) / mag_vT, 0.0f);
            //V[idx] = -muN * velNormal + a * vT;
            //V[idx] = X[idx] - XTilde[idx];
        }
        else
        {
            X[idx] = XTilde[idx];
        }
        //XTilde[idx] = X[idx];
    }

    // Should compute SiTAiTAiSi, which is a sparse matrix
    // Ai here is I
    // size of row, col, val are 48 * numTets + numVerts
    // row, col, val are used to initialize sparse matrix SiTSi
    __global__ void computeSiTSi(int* rowIdx, int* colIdx, float* val, const float* V0, const glm::mat3* DmInvs, const indexType* Tets, float weight, int numTets, int numVerts)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numTets)
        {
            //int posStart = index * 12;
            int v0Ind = Tets[index * 4 + 0] * 3;
            int v1Ind = Tets[index * 4 + 1] * 3;
            int v2Ind = Tets[index * 4 + 2] * 3;
            int v3Ind = Tets[index * 4 + 3] * 3;

            // there are numTets of AiSi in total

            glm::mat4x3 AiSi = glm::transpose(DmInvs[index]) * glm::mat4x3{ 1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -1, -1 };
            glm::mat4x4 SiTAiAiSi = glm::transpose(AiSi) * AiSi;

            int start = index * 48;
            int rowLen = 3 * numVerts;
            for (int i = 0; i < 4; i++)
            {
                int vr = Tets[index * 4 + i] * 3;
                for (int j = 0; j < 4; j++)
                {
                    int vc = Tets[index * 4 + j] * 3;
                    for (int k = 0; k < 3; k++)
                    {
                        setRowColVal(start + (i * 12 + j * 3 + k), rowIdx, colIdx, val, vc + k, vr + k, SiTAiAiSi[j][i] * weight * V0[index]);
                    }
                }
            }
        }
    }

    __global__ void setMDt_2(int* rowIdx, int* colIdx, float* val, int startIndex, float massDt_2, int numVerts)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numVerts)
        {
            int offset = index * 3;
            setRowColVal(startIndex + offset + 0, rowIdx, colIdx, val, offset, offset, massDt_2);
            setRowColVal(startIndex + offset + 1, rowIdx, colIdx, val, offset + 1, offset + 1, massDt_2);
            setRowColVal(startIndex + offset + 2, rowIdx, colIdx, val, offset + 2, offset + 2, massDt_2);
        }
    }

    // dt2_m_1 is dt^2 / mass
    // s(n) = q(n) + dt*v(n) + dt^2 * M^(-1) * fext(n)
    __global__ void computeSn(float* sn, float dt, float dt2_m_1, glm::vec3* pos, glm::vec3* vel, const glm::vec3* force, float* b, float massDt_2, int numVerts)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numVerts)
        {
            int offset = index * 3;
            float3 my_pos{ pos[index].x, pos[index].y, pos[index].z };
            float3 my_vel{ vel[index].x, vel[index].y, vel[index].z };
            float3 my_force{ force[index].x, force[index].y, force[index].z };
            float3 sn_val{ my_pos.x + dt * my_vel.x + dt2_m_1 * my_force.x,
                my_pos.y + dt * my_vel.y + dt2_m_1 * my_force.y,
                my_pos.z + dt * my_vel.z + dt2_m_1 * my_force.z };

            sn[offset + 0] = sn_val.x;
            sn[offset + 1] = sn_val.y;
            sn[offset + 2] = sn_val.z;

            b[offset + 0] = sn_val.x * massDt_2;
            b[offset + 1] = sn_val.y * massDt_2;
            b[offset + 2] = sn_val.z * massDt_2;
        }
    }

    __global__ void computeLocal(const float* V0, const float wi, float* xProj, const glm::mat3* DmInvs, const float* qn, const indexType* Tets, int numTets)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numTets)
        {
            const int v0Ind = Tets[index * 4 + 0] * 3;
            const int v1Ind = Tets[index * 4 + 1] * 3;
            const int v2Ind = Tets[index * 4 + 2] * 3;
            const int v3Ind = Tets[index * 4 + 3] * 3;
            const glm::vec3 v0 = glm::vec3(qn[v0Ind + 0], qn[v0Ind + 1], qn[v0Ind + 2]);
            const glm::vec3 v1 = glm::vec3(qn[v1Ind + 0], qn[v1Ind + 1], qn[v1Ind + 2]);
            const glm::vec3 v2 = glm::vec3(qn[v2Ind + 0], qn[v2Ind + 1], qn[v2Ind + 2]);
            const glm::vec3 v3 = glm::vec3(qn[v3Ind + 0], qn[v3Ind + 1], qn[v3Ind + 2]);
            const glm::mat3 DmInv = DmInvs[index];

            glm::mat3 R = glm::mat3(v0 - v3, v1 - v3, v2 - v3) * DmInv;
            glm::mat3 U;
            glm::mat3 S;

            svdGLM(R, U, S, R);

            R = U * glm::transpose(R);

            if (glm::determinant(R) < 0)
            {
                R[2] = -R[2];
            }

            const glm::mat4x3 piTAiSi = glm::abs(V0[index]) * wi * R * glm::transpose(DmInv)
                * glm::mat4x3{ 1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -1, -1 };

            atomicAdd(&(xProj[v0Ind + 0]), piTAiSi[0][0]);
            atomicAdd(&(xProj[v0Ind + 1]), piTAiSi[0][1]);
            atomicAdd(&(xProj[v0Ind + 2]), piTAiSi[0][2]);

            atomicAdd(&(xProj[v1Ind + 0]), piTAiSi[1][0]);
            atomicAdd(&(xProj[v1Ind + 1]), piTAiSi[1][1]);
            atomicAdd(&(xProj[v1Ind + 2]), piTAiSi[1][2]);

            atomicAdd(&(xProj[v2Ind + 0]), piTAiSi[2][0]);
            atomicAdd(&(xProj[v2Ind + 1]), piTAiSi[2][1]);
            atomicAdd(&(xProj[v2Ind + 2]), piTAiSi[2][2]);

            atomicAdd(&(xProj[v3Ind + 0]), piTAiSi[3][0]);
            atomicAdd(&(xProj[v3Ind + 1]), piTAiSi[3][1]);
            atomicAdd(&(xProj[v3Ind + 2]), piTAiSi[3][2]);
        }
    }

    __global__ void computeM_h2Sn(float* b, float* sn, float massDt_2, int numVerts)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numVerts)
        {
            int offset = index * 3;
            b[offset + 0] = sn[offset + 0] * massDt_2;
            b[offset + 1] = sn[offset + 1] * massDt_2;
            b[offset + 2] = sn[offset + 2] * massDt_2;
        }
    }

    __global__ void addM_h2Sn(float* b, float* masses, int numVerts)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numVerts)
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
}