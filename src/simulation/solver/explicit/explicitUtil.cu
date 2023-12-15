#include <simulation/solver/explicit/explicitSolver.h>
#include <simulation/solver/solverUtil.cuh>
#include <simulation/simulationContext.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

namespace ExplicitUtil
{
    __device__ glm::mat3 Build_Edge_Matrix(const glm::vec3* X, const indexType* Tet, int tet) {
        glm::mat3 ret(0.0f);
        ret[0] = X[Tet[tet * 4 + 1]] - X[Tet[tet * 4]];
        ret[1] = X[Tet[tet * 4 + 2]] - X[Tet[tet * 4]];
        ret[2] = X[Tet[tet * 4 + 3]] - X[Tet[tet * 4]];

        return ret;
    }

    __global__ void computeInvDm(glm::mat3* inv_Dm, int numTets, const glm::vec3* X, const indexType* Tet)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numTets)
        {
            glm::mat3 Dm = Build_Edge_Matrix(X, Tet, index);
            inv_Dm[index] = glm::inverse(Dm);
        }
    }

    __device__ glm::mat3 SaintVenantKirchhoff(const glm::mat3& F, float stiffness_0, float stiffness_1) {
        glm::mat3 U, D, V;
        svdGLM(F, U, D, V);
        float I = D[0][0] * D[0][0] + D[1][1] * D[1][1] + D[2][2] * D[2][2];
        float J = D[0][0] * D[1][1] * D[2][2];
        float II = D[0][0] * D[0][0] * D[0][0] * D[0][0] + D[1][1] * D[1][1] * D[1][1] * D[1][1] + D[2][2] * D[2][2] * D[2][2] * D[2][2];
        float III = J * J;
        float dEdI = stiffness_0 * (I - 3) * 0.25f - stiffness_1 * 0.5f;
        float dEdII = stiffness_1 * 0.25f;
        float dEdIII = 0;

        glm::mat3 P(0.0f);
        P[0][0] = 2 * dEdI * D[0][0] + 4 * dEdII * D[0][0] * D[0][0] * D[0][0] + 2 * dEdIII * III / D[0][0];
        P[1][1] = 2 * dEdI * D[1][1] + 4 * dEdII * D[1][1] * D[1][1] * D[1][1] + 2 * dEdIII * III / D[1][1];
        P[2][2] = 2 * dEdI * D[2][2] + 4 * dEdII * D[2][2] * D[2][2] * D[2][2] + 2 * dEdIII * III / D[2][2];
        return U * P * glm::transpose(V);
    }

    __device__ glm::mat3 NeoHookean_wang(const glm::mat3& F, float stiffness_0, float stiffness_1) {
        glm::mat3 U, D, V;
        svdGLM(F, U, D, V);
        float l0 = D[0][0], l1 = D[1][1], l2 = D[2][2];
        float l0q = l0 * l0;
        float l1q = l1 * l1;
        float l2q = l2 * l2;
        float Ic = l0q + l1q + l2q;

        float l0n23 = powf(l0, -2.0f / 3.0f);
        float l1n23 = powf(l1, -2.0f / 3.0f);
        float l2n23 = powf(l2, -2.0f / 3.0f);
        float lan23 = l0n23 * l1n23 * l2n23;

        float s0_2_lan23 = 2.0f * stiffness_0 * lan23;

        float a0 = l0 * s0_2_lan23;
        float a1 = (2.0f * stiffness_0 * powf(l0, -5.0f / 3.0f) * l1n23 * l2n23 * Ic) / 3.0f;
        float a2 = stiffness_1 / (l0q * l1 * l2);

        float b0 = l1 * s0_2_lan23;
        float b1 = (2.0f * stiffness_0 * powf(l1, -5.0f / 3.0f) * l0n23 * l2n23 * Ic) / 3.0f;
        float b2 = stiffness_1 / (l1q * l0 * l2);

        float c0 = l2 * s0_2_lan23;
        float c1 = (2.0f * stiffness_0 * powf(l2, -5.0f / 3.0f) * l0n23 * l1n23 * Ic) / 3.0f;
        float c2 = stiffness_1 / (l2q * l0 * l1);

        float dw0 = a0 - a1 - a2;
        float dw1 = b0 - b1 - b2;
        float dw2 = c0 - c1 - c2;

        D[0][0] = dw0; D[1][1] = dw1; D[2][2] = dw2;
        return U * D * glm::transpose(V);
    }

    __device__ glm::mat3 NeoHookean_wiki(const glm::mat3& F, float stiffness_0, float stiffness_1) {
        glm::mat3 U, D, V;
        svdGLM(F, U, D, V);
        float l0 = D[0][0], l1 = D[1][1], l2 = D[2][2];
        float J = glm::determinant(F);
        float C1 = 0.5f * stiffness_0;
        float dw0 = C1 * (2.0f * l0 - 2.0f / J * l1 * l2) + stiffness_1 * (J - 1.0f) * l1 * l2;
        float dw1 = C1 * (2.0f * l1 - 2.0f / J * l0 * l2) + stiffness_1 * (J - 1.0f) * l0 * l2;
        float dw2 = C1 * (2.0f * l2 - 2.0f / J * l0 * l1) + stiffness_1 * (J - 1.0f) * l0 * l1;
        D[0][0] = dw0; D[1][1] = dw1; D[2][2] = dw2;
        return U * D * glm::transpose(V);
    }

    __global__ void ComputeForcesSVD(glm::vec3* Force, const glm::vec3* X, const indexType* Tet, int tet_number, const glm::mat3* inv_Dm, float stiffness_0, float stiffness_1) {
        int tet = blockIdx.x * blockDim.x + threadIdx.x;
        if (tet >= tet_number) return;

        glm::mat3 F = Build_Edge_Matrix(X, Tet, tet) * inv_Dm[tet];

        glm::mat3 forces = NeoHookean_wiki(F, stiffness_0, stiffness_1) * glm::transpose(inv_Dm[tet]) * (-1.0f / (6.0f * glm::determinant(inv_Dm[tet])));

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

    __global__ void EulerMethod(glm::vec3* X, glm::vec3* V, const glm::vec3* Force, int numVerts, float mass, float dt) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numVerts) return;

        V[i] += Force[i] / mass * dt;
        X[i] += V[i] * dt;
    }

    __global__ void LaplacianKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int numVerts, const indexType* Tet, float blendAlpha) {
        int i = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (i < numVerts) {
            V[i] = blendAlpha * V[i] + (1 - blendAlpha) * V_sum[i] / float(V_num[i]);
        }
    }

    __global__ void LaplacianGatherKern(glm::vec3* V, glm::vec3* V_sum, int* V_num, int numTets, const indexType* Tet) {
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