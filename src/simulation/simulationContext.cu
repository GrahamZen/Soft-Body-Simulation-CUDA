#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <sceneStructs.h>
#include <simulationContext.h>
#include <utilities.cuh>
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include <bvh.cuh>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// TODO: static variables for device memory, any extra info you need, etc
// ...

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */

SimulationCUDAContext::~SimulationCUDAContext()
{
    cudaFree(dev_Xs);
    cudaFree(dev_Tets);
    cudaFree(dev_Vs);
    cudaFree(dev_Fs);
    cudaFree(dev_X0s);
    cudaFree(dev_XTilts);
    for (auto softbody : softBodies) {
        delete softbody;
    }
}

AABB SimulationCUDAContext::GetAABB() const
{
    thrust::device_ptr<glm::vec3> dev_ptr(dev_Xs);
    thrust::device_ptr<glm::vec3> dev_ptrTilts(dev_XTilts);
    return computeBoundingBox(dev_ptr, dev_ptr + numVerts).expand(computeBoundingBox(dev_ptrTilts, dev_ptrTilts + numVerts));
}

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
    int blocks = (softBodyData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    TransformVertices << < blocks, threadsPerBlock >> > (softBodyData.dev_X, model, softBodyData.numVerts);

    auto idx = loadEleFile(eleFileName, startIndex, softBodyData.numTets);
    cudaMalloc((void**)&softBodyData.Tets, sizeof(GLuint) * idx.size());
    cudaMemcpy(softBodyData.Tets, idx.data(), sizeof(GLuint) * idx.size(), cudaMemcpyHostToDevice);
    totalNumVerts += softBodyData.numVerts;
    totalNumTets += softBodyData.numTets;

    m_softBodyData.push_back({ softBodyData, attrib });
}

void DataLoader::AllocData(std::vector<int>& startIndices, glm::vec3*& gX, glm::vec3*& gX0, glm::vec3*& gXTilt, glm::vec3*& gV, glm::vec3*& gF, GLuint*& gTet, int& numVerts, int& numTets)
{
    numVerts = totalNumVerts;
    numTets = totalNumTets;
    cudaMalloc((void**)&gX, sizeof(glm::vec3) * totalNumVerts);
    cudaMalloc((void**)&gX0, sizeof(glm::vec3) * totalNumVerts);
    cudaMalloc((void**)&gXTilt, sizeof(glm::vec3) * totalNumVerts);
    cudaMalloc((void**)&gV, sizeof(glm::vec3) * totalNumVerts);
    cudaMalloc((void**)&gF, sizeof(glm::vec3) * totalNumVerts);
    cudaMemset(gV, 0, sizeof(glm::vec3) * totalNumVerts);
    cudaMemset(gF, 0, sizeof(glm::vec3) * totalNumVerts);
    cudaMalloc((void**)&gTet, sizeof(GLuint) * totalNumTets * 4);
    int vertOffset = 0, tetOffset = 0;
    thrust::device_ptr<GLuint> dev_ptr(gTet);
    for (auto& softBodyData : m_softBodyData)
    {
        startIndices.push_back(vertOffset);
        auto& data = softBodyData.first;
        cudaMemcpy(gX + vertOffset, data.dev_X, sizeof(glm::vec3) * data.numVerts, cudaMemcpyDeviceToDevice);
        thrust::transform(data.Tets, data.Tets + data.numTets * 4, dev_ptr + tetOffset, [vertOffset] __device__(GLuint x) {
            return x + vertOffset;
        });
        cudaFree(data.dev_X);
        data.dev_X = gX + vertOffset;
        data.dev_X0 = gX0 + vertOffset;
        data.dev_XTilt = gXTilt + vertOffset;
        data.dev_V = gV + vertOffset;
        data.dev_F = gF + vertOffset;
        vertOffset += data.numVerts;
        tetOffset += data.numTets * 4;
    }
    cudaMemcpy(gX0, gX, sizeof(glm::vec3) * totalNumVerts, cudaMemcpyDeviceToDevice);
    cudaMemcpy(gXTilt, gX, sizeof(glm::vec3) * totalNumVerts, cudaMemcpyDeviceToDevice);
}

void SimulationCUDAContext::CCD()
{
    float* tIs = m_bvh.DetectCollisionCandidates(dev_Tets, dev_Xs, dev_XTilts, dev_TetIds);
    int blocks = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    //CCDKernel << <blocks, threadsPerBlock >> > (dev_Xs, dev_XTilts, tIs, numVerts);
}

void SimulationCUDAContext::Update()
{
    for (auto softbody : softBodies) {
        softbody->Update();
    }
    m_bvh.BuildBVHTree(GetAABB(), numTets, dev_Xs, dev_XTilts, dev_Tets);
    glm::vec3 floorPos = glm::vec3(0.0f, -4.0f, 0.0f);
    glm::vec3 floorUp = glm::vec3(0.0f, 1.0f, 0.0f);
    HandleFloorCollision << <(numVerts + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (dev_XTilts, dev_Vs, numVerts, floorPos, floorUp, muT, muN);
    //CCD();
    cudaMemcpy(dev_Xs, dev_XTilts, sizeof(glm::vec3) * numVerts, cudaMemcpyDeviceToDevice);
    if (context->guiData->ObjectVis) {
        PrepareRenderData();
    }
    if (context->guiData->BVHVis)
        m_bvh.PrepareRenderData();
}

void SimulationCUDAContext::PrepareRenderData() {
    for (auto softbody : softBodies) {
        glm::vec3* pos;
        glm::vec4* nor;
        softbody->Mesh::mapDevicePtr(&pos, &nor);
        dim3 numThreadsPerBlock(softbody->getTetNumber() / threadsPerBlock + 1);

        PopulatePos << <numThreadsPerBlock, threadsPerBlock >> > (pos, softbody->getX(), softbody->getTet(), softbody->getTetNumber());
        RecalculateNormals << <softbody->getTetNumber() * 4 / threadsPerBlock + 1, threadsPerBlock >> > (nor, pos, 4 * softbody->getTetNumber());
        softbody->Mesh::unMapDevicePtr();
    }
}


