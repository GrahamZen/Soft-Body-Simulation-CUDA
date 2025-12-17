#include <def.h>
#include <svd.cuh>
#include <simulation/solver/projective/pdUtil.cuh>

namespace PdUtil {
    // Should compute SiTAiTAiSi, which is a sparse matrix
    // Ai here is I
    // size of row, col, val are 48 * numTets + numVerts
    // row, col, val are used to initialize sparse matrix SiTSi
    __global__ void computeSiTSi(int* rowIdx, int* colIdx, float* val, float* matrix_diag, const float* V0, const glm::mat3* DmInvs, const indexType* Tets, const float* weight, int numTets, int numVerts)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numTets)
        {
            // there are numTets of AiSi in total

            glm::mat4x3 AiSi = glm::transpose(DmInvs[index]) * glm::mat4x3{ -1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1 };
            glm::mat4x4 SiTAiAiSi = glm::transpose(AiSi) * AiSi;

            const float coef = V0[index] * weight[index];

            atomicAdd(&matrix_diag[Tets[index * 4 + 0]], SiTAiAiSi[0][0] * coef);
            atomicAdd(&matrix_diag[Tets[index * 4 + 1]], SiTAiAiSi[1][1] * coef);
            atomicAdd(&matrix_diag[Tets[index * 4 + 2]], SiTAiAiSi[2][2] * coef);
            atomicAdd(&matrix_diag[Tets[index * 4 + 3]], SiTAiAiSi[3][3] * coef);

            int start = index * 48;
            for (int i = 0; i < 4; i++)
            {
                int vr = Tets[index * 4 + i] * 3;
                for (int j = 0; j < 4; j++)
                {
                    int vc = Tets[index * 4 + j] * 3;
                    for (int k = 0; k < 3; k++)
                    {
                        setRowColVal(start + (i * 12 + j * 3 + k), rowIdx, colIdx, val, vc + k, vr + k, SiTAiAiSi[j][i] * coef);
                    }
                }
            }
        }
    }

    __global__ void setMDt_2(int numVerts, int* rowIdx, int* colIdx, float* val, int offset, const float* masses, float dt2, float* massDt_2s, float* DBC, float weight)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numVerts)
        {
            int start = index * 3;
            float massDt_2 = (masses[index] + DBC[index] * weight) / dt2;
            massDt_2s[index] = massDt_2;
            setRowColVal(offset + start + 0, rowIdx, colIdx, val, start, start, massDt_2);
            setRowColVal(offset + start + 1, rowIdx, colIdx, val, start + 1, start + 1, massDt_2);
            setRowColVal(offset + start + 2, rowIdx, colIdx, val, start + 2, start + 2, massDt_2);
        }
    }

    __global__ void setMDt_2MoreDBC(int numVerts, const float* masses, float dt2, float* massDt_2s, float* moreDBC, float* DBC)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numVerts && DBC[index] == 0)
        {
            float wi = moreDBC[index];
            if (wi > 0) {
                massDt_2s[index] = (masses[index] + wi) / dt2;
            }
            else {
                massDt_2s[index] = masses[index] / dt2;
            }
        }
    }

    // dt2_m_1 is dt^2 / mass
    // s(n) = q(n) + dt*v(n) + dt^2 * M^(-1) * fext(n)
    __global__ void computeSn(int numVerts, float* sn, float dt, const float* massDt_2s, glm::vec3* pos, glm::vec3* vel, const glm::vec3* force, const float* more_fixed, const glm::vec3* offset_X, glm::vec3* fixed_X, glm::vec3 dir)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numVerts)
        {
            if (more_fixed[index]) {
                fixed_X[index] = offset_X[index] + dir;
            }

            int offset = index * 3;
            float massDt_2 = massDt_2s[index];
            float dt2_m_1 = 1 / massDt_2;

            float3 my_pos{ pos[index].x, pos[index].y, pos[index].z };
            float3 my_vel{ vel[index].x, vel[index].y, vel[index].z };
            float3 my_force{ force[index].x, force[index].y, force[index].z };
            float3 sn_val{ my_pos.x + dt * my_vel.x + dt2_m_1 * my_force.x,
                my_pos.y + dt * my_vel.y + dt2_m_1 * my_force.y,
                my_pos.z + dt * my_vel.z + dt2_m_1 * my_force.z };

            sn[offset + 0] = sn_val.x;
            sn[offset + 1] = sn_val.y;
            sn[offset + 2] = sn_val.z;
        }
    }

    __global__ void computeLocal(const float* V0, const float* wi, float* xProj, const glm::mat3* DmInvs, const float* qn, const indexType* Tets, int numTets, bool isJacobi)
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

            glm::mat3 F = glm::mat3(v1 - v0, v2 - v0, v3 - v0) * DmInv;
            glm::mat3 U, S, R;

            svdGLM(F, U, S, R);

            R = U * glm::transpose(R);

            if (glm::determinant(R) < 0)
            {
                R[2] = -R[2];
            }
            if (isJacobi) {
                R = R - F;
            }
            glm::mat4x3 piTAiSi = glm::abs(V0[index]) * wi[index] * R * glm::transpose(DmInv)
                * glm::mat4x3{ -1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1 };

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

    __global__ void computeDBCLocal(int numVerts, float* DBC, float* moreDBC, const glm::vec3* x0, const float wi, float* xProj)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numVerts)
        {
            float moreWi = moreDBC[index];
            if (DBC[index] > 0)
            {
                xProj[index * 3 + 0] = x0[index].x * wi;
                xProj[index * 3 + 1] = x0[index].y * wi;
                xProj[index * 3 + 2] = x0[index].z * wi;
            }
            else if (moreWi > 0)
            {
                xProj[index * 3 + 0] = x0[index].x * moreWi;
                xProj[index * 3 + 1] = x0[index].y * moreWi;
                xProj[index * 3 + 2] = x0[index].z * moreWi;
            }
        }
    }

    __global__ void addM_h2Sn(float* b, float* sn, float* massDt_2s, int numVerts)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numVerts)
        {
            int offset = index * 3;
            float massDt_2 = massDt_2s[index];
            b[offset + 0] = massDt_2 * sn[offset + 0];
            b[offset + 1] = massDt_2 * sn[offset + 1];
            b[offset + 2] = massDt_2 * sn[offset + 2];
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

    __global__ void getErrorKern(int numVerts, float* next_x, const float* b, const float* massDt_2s, const float* sn, const float* matrix_diag)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numVerts)
        {
            float c = massDt_2s[index];
            float md = matrix_diag[index];
            next_x[index * 3 + 0] = (b[index * 3 + 0] - c * sn[index * 3 + 0]) / (c + md) + sn[index * 3 + 0];
            next_x[index * 3 + 1] = (b[index * 3 + 1] - c * sn[index * 3 + 1]) / (c + md) + sn[index * 3 + 1];
            next_x[index * 3 + 2] = (b[index * 3 + 2] - c * sn[index * 3 + 2]) / (c + md) + sn[index * 3 + 2];
        }
    }

    __global__ void chebyshevKern(int numVerts3, float* next_x, float* prev_x, float* sn, float omega)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numVerts3)
        {
            next_x[index] = 0.9 * (next_x[index] - sn[index]) + sn[index];
            next_x[index] = (next_x[index] - prev_x[index]) * omega + prev_x[index];
            prev_x[index] = sn[index];
            sn[index] = next_x[index];
        }
    }
}