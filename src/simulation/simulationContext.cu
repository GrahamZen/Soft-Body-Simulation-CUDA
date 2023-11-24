#include <cuda.h>

#include <sceneStructs.h>
#include <simulationContext.h>
#include <utilities.cuh>
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// TODO: static variables for device memory, any extra info you need, etc
// ...

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */

void DataLoader::CollectData(const char* nodeFileName, const char* eleFileName, const glm::vec3& pos, const glm::vec3& scale, const glm::vec3& rot,
    bool centralize, int startIndex, SoftBody::SoftBodyAttribute attrib)
{
    SoftBodyData softBodyData;
    auto vertices = loadNodeFile(nodeFileName, centralize, softBodyData.numVerts);
    cudaMalloc((void**)&softBodyData.dev_X, sizeof(glm::vec3) * softBodyData.numVerts);
    cudaMemcpy(softBodyData.dev_X, vertices.data(), sizeof(glm::vec3) * softBodyData.numVerts, cudaMemcpyHostToDevice);

    // transform
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, pos);
    model = glm::scale(model, scale);
    model = glm::rotate(model, glm::radians(rot.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
    int threadsPerBlock = 64;
    int blocks = (softBodyData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    TransformVertices << < blocks, threadsPerBlock >> > (softBodyData.dev_X, model, softBodyData.numVerts);

    auto idx = loadEleFile(eleFileName, startIndex, softBodyData.numTets);
    cudaMalloc((void**)&softBodyData.Tets, sizeof(GLuint) * idx.size());
    cudaMemcpy(softBodyData.Tets, idx.data(), sizeof(GLuint) * idx.size(), cudaMemcpyHostToDevice);
    totalNumVerts += softBodyData.numVerts;

    m_softBodyData.push_back({ softBodyData, attrib });
}

glm::vec3* DataLoader::AllocData(std::vector<int>& startIndices)
{
    cudaMalloc((void**)&dev_XPtr, sizeof(glm::vec3) * totalNumVerts);
    int offset = 0;
    for (auto& softBodyData : m_softBodyData)
    {
        startIndices.push_back(offset);
        auto& data = softBodyData.first;
        cudaMemcpy(dev_XPtr + offset, data.dev_X, sizeof(glm::vec3) * data.numVerts, cudaMemcpyDeviceToDevice);
        cudaFree(data.dev_X);
        data.dev_X = dev_XPtr + offset;
        offset += data.numVerts;
    }
    return dev_XPtr;
}

void SimulationCUDAContext::Update()
{
    //m_bvh.BuildBVHTree(0, GetAABB(), GetTetCnt(), softBodies);
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