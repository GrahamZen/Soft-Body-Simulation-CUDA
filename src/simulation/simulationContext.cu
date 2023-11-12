#include <cuda.h>

#include <sceneStructs.h>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include <simulationContext.h>
#include <utilities.h>
#include <utilities.cuh>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

static GuiDataContainer* guiData = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
SimulationCUDAContext::SimulationCUDAContext()
{
}

SimulationCUDAContext::~SimulationCUDAContext()
{
    for (auto softbody : softBodies) {
        delete softbody;
    }
}

void SimulationCUDAContext::Update()
{
    for (auto softbody : softBodies) {
        softbody->Update();
        glm::vec3* pos;
        glm::vec4* nor;
        softbody->mapDevicePtr(&pos, &nor);
        dim3 numThreadsPerBlock(softbody->getTetNumber() / 32 + 1);
        PopulatePos << <numThreadsPerBlock, 32 >> > (pos, softbody->getX(), softbody->getTet(), softbody->getTetNumber());
        RecalculateNormals << <softbody->getTetNumber() * 4 / 32 + 1, 32 >> > (nor, pos, 4 * softbody->getTetNumber());
        softbody->unMapDevicePtr();
    }
}

SoftBody::SoftBody(const char* nodeFileName, const char* eleFileName) :Mesh()
{
    std::vector<glm::vec3> vertices = loadNodeFile(nodeFileName);
    number = vertices.size();
    cudaMalloc((void**)&X, sizeof(glm::vec3) * number);
    cudaMemcpy(X, vertices.data(), sizeof(glm::vec3) * number, cudaMemcpyHostToDevice);


    std::vector<GLuint> idx = loadEleFile(eleFileName);
    tet_number = idx.size() / 4;
    cudaMalloc((void**)&Tet, sizeof(GLuint) * idx.size());
    cudaMemcpy(Tet, idx.data(), sizeof(GLuint) * idx.size(), cudaMemcpyHostToDevice);

    Mesh::tet_number = tet_number;

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
    int threadsPerBlock = 64;
    int blocks = (tet_number + threadsPerBlock - 1) / threadsPerBlock;
    computeInvDm << < blocks, threadsPerBlock >> > (inv_Dm, tet_number, X, Tet);
}

SoftBody::~SoftBody()
{
    cudaFree(X);
    cudaFree(Tet);
    cudaFree(Force);
    cudaFree(V);
    cudaFree(inv_Dm);
    cudaFree(V_sum);
}

void SoftBody::mapDevicePtr(glm::vec3** bufPosDevPtr, glm::vec4** bufNorDevPtr)
{
    size_t size;
    cudaGraphicsMapResources(1, &cuda_bufPos_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)bufPosDevPtr, &size, cuda_bufPos_resource);

    cudaGraphicsMapResources(1, &cuda_bufNor_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)bufNorDevPtr, &size, cuda_bufNor_resource);
}

void SoftBody::unMapDevicePtr()
{
    cudaGraphicsUnmapResources(1, &cuda_bufPos_resource, 0);
    cudaGraphicsUnmapResources(1, &cuda_bufNor_resource, 0);
}

void SoftBody::Laplacian_Smoothing(float blendAlpha)
{
    cudaMemset(V_sum, 0, sizeof(glm::vec3) * number);
    cudaMemset(V_num, 0, sizeof(int) * number);
    int threadsPerBlock = 64;
    int blocks = (tet_number + threadsPerBlock - 1) / threadsPerBlock;
    LaplacianGatherKern << < blocks, threadsPerBlock >> > (V, V_sum, V_num, tet_number, Tet);
    LaplacianKern << < (number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (V, V_sum, V_num, number, Tet, blendAlpha);
}

void SoftBody::Update()
{
    for (int l = 0; l < 10; l++)
        _Update();
}

void SoftBody::_Update()
{
    int threadsPerBlock = 64;
    AddGravity << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (Force, V, mass, number, jump);
    Laplacian_Smoothing();
    glm::vec3 floorPos = glm::vec3(0.0f, -4.0f, 0.0f);
    glm::vec3 floorUp = glm::vec3(0.0f, 1.0f, 0.0f);
    // ComputeForces << <(tet_number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (Force, X, Tet, tet_number, inv_Dm, stiffness_0, stiffness_1);
    UpdateParticles << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (X, V, Force, number, mass, dt, damp, floorPos, floorUp, muT, muN);
}
