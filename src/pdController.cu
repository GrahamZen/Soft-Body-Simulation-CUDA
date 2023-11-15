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

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

float* dev_sn;
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

// dt2_m_1 is dt^2 / mass
// s(n) = q(n) + dt*v(n) + dt^2 * M^(-1) * fext(n)
__global__ void computeSn(float* sn, float dt, float dt2_m_1, glm::vec3* pos, glm::vec3* vel, glm::vec3* force, int numVerts)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numVerts)
    {
        sn[index * 3 + 0] = pos[index].x + dt * vel[index].x + dt2_m_1 * force[index].x;
        sn[index * 3 + 1] = pos[index].y + dt * vel[index].y + dt2_m_1 * force[index].y;
        sn[index * 3 + 2] = pos[index].z + dt * vel[index].z + dt2_m_1 * force[index].z;
    }
}

__device__ void computeLocalDm(glm::mat3* DmInv, float* qn__1, int* tetIndex, int tetNumber)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < tetNumber)
    {
        glm::vec3 v0 = glm::vec3(qn__1[index * 12 + 0], qn__1[index * 12 + 1], qn__1[index * 12 + 2]);
        glm::vec3 v1 = glm::vec3(qn__1[index * 12 + 3], qn__1[index * 12 + 4], qn__1[index * 12 + 5]);
        glm::vec3 v2 = glm::vec3(qn__1[index * 12 + 6], qn__1[index * 12 + 7], qn__1[index * 12 + 8]);
        glm::vec3 v3 = glm::vec3(qn__1[index * 12 + 9], qn__1[index * 12 + 10], qn__1[index * 12 + 11]);

        glm::mat3 Dl;
        Dl[0] = v0 - v3;
        Dl[1] = v1 - v3;
        Dl[2] = v2 - v3;
        
        glm::mat3 F = Dl * DmInv[index];
        glm::mat3 U;
        glm::mat3 S;
        glm::mat3 V;
        svd(F,U,S,V);

        glm::mat3 Fstar = U * S * glm::transpose(V);
        
    }
}

// project explicit q onto the constraint
void solveLocal()
{

}

void solveGlobal()
{

}


void pdStep()
{

}