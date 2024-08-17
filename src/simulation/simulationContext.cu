#include <utilities.cuh>
#include <simulation/simulationContext.h>
#include <simulation/softBody.h>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <simulation/MshLoader.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// TODO: static variables for device memory, any extra info you need, etc
// ...

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */

__global__ void CCDKernel(glm::vec3* X, glm::vec3* XTilt, glm::vec3* V, dataType* tI, glm::vec3* normal, float muT, float muN, int numVerts);

SimulationCUDAContext::~SimulationCUDAContext()
{
    for (auto name : namesSoftBodies) {
        delete[]name;
    }
    cudaFree(mSolverData.X);
    cudaFree(mSolverData.Tet);
    cudaFree(mSolverData.V);
    cudaFree(mSolverData.Force);
    cudaFree(mSolverData.X0);
    cudaFree(mSolverData.XTilt);
    for (auto softbody : softBodies) {
        delete softbody;
    }
    cudaFree(dev_Normals);
    delete mSolver;
}

int SimulationCUDAContext::GetVertCnt() const {
    return mSolverData.numVerts;
}

int SimulationCUDAContext::GetNumQueries() const {
    return mCollisionDetection.GetNumQueries();
}

int SimulationCUDAContext::GetTetCnt() const {
    return mSolverData.numTets;
}

void DataLoader::CollectData(const char* nodeFileName, const char* eleFileName, const char* faceFileName, const glm::vec3& pos, const glm::vec3& scale, const glm::vec3& rot,
    bool centralize, int startIndex, SolverAttribute attrib)
{
    SolverData softBodyData;
    auto vertices = loadNodeFile(nodeFileName, centralize, softBodyData.numVerts);
    cudaMalloc((void**)&softBodyData.X, sizeof(glm::vec3) * softBodyData.numVerts);
    cudaMemcpy(softBodyData.X, vertices.data(), sizeof(glm::vec3) * softBodyData.numVerts, cudaMemcpyHostToDevice);

    // transform
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, pos);
    model = glm::scale(model, scale);
    model = glm::rotate(model, glm::radians(rot.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
    int blocks = (softBodyData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    TransformVertices << < blocks, threadsPerBlock >> > (softBodyData.X, model, softBodyData.numVerts);

    auto tetIdx = loadEleFile(eleFileName, startIndex, softBodyData.numTets);
    cudaMalloc((void**)&softBodyData.Tet, sizeof(indexType) * tetIdx.size());
    cudaMemcpy(softBodyData.Tet, tetIdx.data(), sizeof(indexType) * tetIdx.size(), cudaMemcpyHostToDevice);
    auto triIdx = loadFaceFile(faceFileName, startIndex, softBodyData.numTris);
    if (!triIdx.empty()) {
        cudaMalloc((void**)&softBodyData.Tri, sizeof(indexType) * triIdx.size());
        cudaMemcpy(softBodyData.Tri, triIdx.data(), sizeof(indexType) * triIdx.size(), cudaMemcpyHostToDevice);
    }
    else {
        softBodyData.Tri = nullptr;
        softBodyData.numTris = 0;
    }
    CollectEdges(triIdx);
    totalNumVerts += softBodyData.numVerts;
    totalNumTets += softBodyData.numTets;

    m_softBodyData.push_back({ softBodyData, attrib });
}

void DataLoader::CollectData(const char* mshFileName, const glm::vec3& pos, const glm::vec3& scale, const glm::vec3& rot,
    bool centralize, int startIndex, SolverAttribute attrib)
{
    SolverData softBodyData;
    igl::MshLoader _loader(mshFileName);
    auto nodes = _loader.get_nodes();
    std::vector<float> vertices(nodes.size());
    softBodyData.numVerts = nodes.size() / 3;
    std::transform(nodes.begin(), nodes.end(), vertices.begin(), [](igl::MshLoader::Float f) {
        return static_cast<float>(f);
        });
    cudaMalloc((void**)&softBodyData.X, sizeof(glm::vec3) * softBodyData.numVerts);
    cudaMemcpy(softBodyData.X, vertices.data(), sizeof(glm::vec3) * softBodyData.numVerts, cudaMemcpyHostToDevice);

    // transform
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, pos);
    model = glm::scale(model, scale);
    model = glm::rotate(model, glm::radians(rot.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
    int blocks = (softBodyData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    TransformVertices << < blocks, threadsPerBlock >> > (softBodyData.X, model, softBodyData.numVerts);

    auto elements = _loader.get_elements();
    std::vector<indexType> tetIdx(elements.size());
    std::transform(elements.begin(), elements.end(), tetIdx.begin(), [](int i) {
        return static_cast<indexType>(i);
        });
    softBodyData.numTets = tetIdx.size() / 4;
    cudaMalloc((void**)&softBodyData.Tet, sizeof(indexType) * tetIdx.size());
    cudaMemcpy(softBodyData.Tet, tetIdx.data(), sizeof(indexType) * tetIdx.size(), cudaMemcpyHostToDevice);
    std::vector<indexType> triIdx;
    if (!triIdx.empty()) {
        cudaMalloc((void**)&softBodyData.Tri, sizeof(indexType) * triIdx.size());
        cudaMemcpy(softBodyData.Tri, triIdx.data(), sizeof(indexType) * triIdx.size(), cudaMemcpyHostToDevice);
    }
    else {
        softBodyData.Tri = nullptr;
        softBodyData.numTris = 0;
    }
    CollectEdges(triIdx);
    totalNumVerts += softBodyData.numVerts;
    totalNumTets += softBodyData.numTets;

    m_softBodyData.push_back({ softBodyData, attrib });
}

void DataLoader::AllocData(std::vector<int>& startIndices, glm::vec3*& gX, glm::vec3*& gX0, glm::vec3*& gXTilt,
    glm::vec3*& gV, glm::vec3*& gF, indexType*& gEdges, indexType*& gTet, indexType*& gTetFather, int& numVerts, int& numTets)
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
    cudaMalloc((void**)&gEdges, sizeof(indexType) * totalNumEdges * 2);
    cudaMalloc((void**)&gTet, sizeof(indexType) * totalNumTets * 4);
    cudaMalloc((void**)&gTetFather, sizeof(indexType) * totalNumTets);
    int vertOffset = 0, tetOffset = 0, edgeOffset = 0;
    thrust::device_ptr<indexType> dev_gTetPtr(gTet);
    thrust::device_ptr<indexType> dev_gEdgesPtr(gEdges);
    thrust::device_ptr<indexType> dev_gTetFatherPtr(gTetFather);
    for (int i = 0; i < m_softBodyData.size(); i++)
    {
        auto& softBodyData = m_softBodyData[i];
        startIndices.push_back(vertOffset);
        auto& data = softBodyData.first;
        cudaMemcpy(gX + vertOffset, data.X, sizeof(glm::vec3) * data.numVerts, cudaMemcpyDeviceToDevice);
        thrust::transform(data.Tet, data.Tet + data.numTets * 4, dev_gTetPtr + tetOffset, [vertOffset] __device__(indexType x) {
            return x + vertOffset;
        });
        thrust::fill(dev_gTetFatherPtr + tetOffset / 4, dev_gTetFatherPtr + tetOffset / 4 + data.numTets, i);
        cudaMemcpy(gEdges + edgeOffset, m_edges[i].data(), sizeof(indexType) * m_edges[i].size(), cudaMemcpyHostToDevice);
        thrust::transform(dev_gEdgesPtr + edgeOffset, dev_gEdgesPtr + edgeOffset + m_edges[i].size(), dev_gEdgesPtr + edgeOffset,
            [vertOffset] __device__(indexType x) {
            return x + vertOffset;
        });
        cudaFree(data.X);
        data.X = gX + vertOffset;
        data.X0 = gX0 + vertOffset;
        data.XTilt = gXTilt + vertOffset;
        data.V = gV + vertOffset;
        data.Force = gF + vertOffset;
        vertOffset += data.numVerts;
        tetOffset += data.numTets * 4;
        edgeOffset += m_edges[i].size();
    }
    cudaMemcpy(gX0, gX, sizeof(glm::vec3) * totalNumVerts, cudaMemcpyDeviceToDevice);
    cudaMemcpy(gXTilt, gX, sizeof(glm::vec3) * totalNumVerts, cudaMemcpyDeviceToDevice);
}

void SimulationCUDAContext::CCD()
{
    mCollisionDetection.DetectCollision(dev_tIs, dev_Normals);
    int blocks = (mSolverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    CCDKernel << <blocks, threadsPerBlock >> > (mSolverData.X, mSolverData.XTilt, mSolverData.V, dev_tIs, dev_Normals, mSolverParams.muT, mSolverParams.muN, mSolverData.numVerts);
}

void SimulationCUDAContext::PrepareRenderData() {
    for (auto softbody : softBodies) {
        glm::vec3* pos;
        glm::vec4* nor;
        softbody->Mesh::MapDevicePtr(&pos, &nor);
        if (softbody->GetNumTris() == 0) {
            dim3 numThreadsPerBlock(softbody->GetNumTets() / threadsPerBlock + 1);
            PopulatePos << <numThreadsPerBlock, threadsPerBlock >> > (pos, softbody->GetSolverData().X, softbody->GetSolverData().Tet, softbody->GetNumTets());
            RecalculateNormals << <softbody->GetNumTets() * 4 / threadsPerBlock + 1, threadsPerBlock >> > (nor, pos, 4 * softbody->GetNumTets());
            softbody->Mesh::UnMapDevicePtr();
        }
        else {
            dim3 numThreadsPerBlock(softbody->GetNumTris() / threadsPerBlock + 1);
            PopulateTriPos << <numThreadsPerBlock, threadsPerBlock >> > (pos, softbody->GetSolverData().X, softbody->GetSolverData().Tri, softbody->GetNumTris());
            RecalculateNormals << <softbody->GetNumTris() / threadsPerBlock + 1, threadsPerBlock >> > (nor, pos, softbody->GetNumTris());
            softbody->Mesh::UnMapDevicePtr();
        }
    }
}