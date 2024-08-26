#include <utilities.cuh>
#include <simulation/solver/IPC/ipc.h>
#include <simulation/softBody.h>
#include <simulation/dataLoader.h>
#include <simulation/MshLoader.h>
#include <simulation/simulationContext.h>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/fill.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// TODO: static variables for device memory, any extra info you need, etc
// ...

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */

SimulationCUDAContext::SimulationCUDAContext(Context* ctx, const std::string& _name, nlohmann::json& json,
    const std::map<std::string, nlohmann::json>& softBodyDefs, std::vector<FixedBody*>& _fixedBodies, int _threadsPerBlock, int _threadsPerBlockBVH, int maxThreads, int _numIterations)
    :context(ctx), threadsPerBlock(_threadsPerBlock), fixedBodies(_fixedBodies), name(_name)
{
    DataLoader<double> dataLoader(threadsPerBlock);
    mSolverParams.pCollisionDetection = new CollisionDetection{ this, _threadsPerBlockBVH, 1 << 16 };
    if (json.contains("dt")) {
        mSolverParams.dt = json["dt"].get<float>();
    }
    if (json.contains("gravity")) {
        mSolverParams.gravity = json["gravity"].get<float>();
    }
    if (json.contains("pause")) {
        context->guiData->Pause = json["pause"].get<bool>();
    }
    if (json.contains("damp")) {
        float damp = json["damp"].get<float>();
    }
    if (json.contains("muN")) {
        float muN = json["muN"].get<float>();
    }
    if (json.contains("muT")) {
        float muT = json["muT"].get<float>();
    }
    if (json.contains("softBodies")) {
        for (const auto& sbJson : json["softBodies"]) {
            auto& sbDefJson = softBodyDefs.at(std::string(sbJson["name"]));
            std::string nodeFile;
            std::string mshFile;
            std::string eleFile;
            if (sbDefJson.contains("nodeFile")) {
                nodeFile = sbDefJson["nodeFile"];
                eleFile = sbDefJson["eleFile"];
            }
            if (sbDefJson.contains("mshFile")) {
                mshFile = sbDefJson["mshFile"];
            }
            std::string faceFile;
            if (sbDefJson.contains("faceFile")) {
                faceFile = sbDefJson["faceFile"];
            }
            glm::vec3 pos;
            glm::vec3 scale;
            glm::vec3 rot;
            float mass;
            float stiffness_0;
            float stiffness_1;
            int constraints;
            if (!sbJson.contains("pos")) {
                if (sbDefJson.contains("pos")) {
                    pos = glm::vec3(sbDefJson["pos"][0].get<float>(), sbDefJson["pos"][1].get<float>(), sbDefJson["pos"][2].get<float>());
                }
                else {
                    pos = glm::vec3(0.f);
                }
            }
            else {
                pos = glm::vec3(sbJson["pos"][0].get<float>(), sbJson["pos"][1].get<float>(), sbJson["pos"][2].get<float>());
            }
            if (!sbJson.contains("scale")) {
                if (sbDefJson.contains("scale")) {
                    scale = glm::vec3(sbDefJson["scale"][0].get<float>(), sbDefJson["scale"][1].get<float>(), sbDefJson["scale"][2].get<float>());
                }
                else {
                    scale = glm::vec3(1.f);
                }
            }
            else {
                scale = glm::vec3(sbJson["scale"][0].get<float>(), sbJson["scale"][1].get<float>(), sbJson["scale"][2].get<float>());
            }
            if (!sbJson.contains("rot")) {
                if (sbDefJson.contains("rot")) {
                    rot = glm::vec3(sbDefJson["rot"][0].get<float>(), sbDefJson["rot"][1].get<float>(), sbDefJson["rot"][2].get<float>());
                }
                else {
                    rot = glm::vec3(0.f);
                }
            }
            else {
                rot = glm::vec3(sbJson["rot"][0].get<float>(), sbJson["rot"][1].get<float>(), sbJson["rot"][2].get<float>());
            }
            if (!sbJson.contains("mass")) {
                mass = sbDefJson["mass"].get<float>();
            }
            else {
                mass = sbJson["mass"].get<float>();
            }
            if (!sbJson.contains("stiffness_0")) {
                stiffness_0 = sbDefJson["stiffness_0"].get<float>();
            }
            else {
                stiffness_0 = sbJson["stiffness_0"].get<float>();
            }
            if (!sbJson.contains("stiffness_1")) {
                stiffness_1 = sbDefJson["stiffness_1"].get<float>();
            }
            else {
                stiffness_1 = sbJson["stiffness_1"].get<float>();
            }
            if (!sbJson.contains("constraints")) {
                constraints = sbDefJson["constraints"].get<int>();
            }
            else {
                constraints = sbJson["constraints"].get<int>();
            }
            bool centralize = sbDefJson["centralize"].get<bool>();
            int startIndex = sbDefJson["start index"].get<int>();
            if (!mshFile.empty()) {
                std::string baseName = mshFile.substr(nodeFile.find_last_of('/') + 1);
                char* name = new char[baseName.size() + 1];
                strcpy(name, baseName.c_str());
                namesSoftBodies.push_back(name);
                dataLoader.CollectData(mshFile.c_str(), pos, scale, rot, centralize, startIndex,
                    SolverAttribute{ mass, stiffness_0, stiffness_1, constraints });
            }
            else if (!nodeFile.empty()) {
                std::string baseName = nodeFile.substr(nodeFile.find_last_of('/') + 1);
                char* name = new char[baseName.size() + 1];
                strcpy(name, baseName.c_str());
                namesSoftBodies.push_back(name);
                dataLoader.CollectData(nodeFile.c_str(), eleFile.c_str(), faceFile.c_str(), pos, scale, rot, centralize, startIndex,
                    SolverAttribute{ mass, stiffness_0, stiffness_1, constraints });
            }
            else {
                throw std::runtime_error("Msh or node file must be provided!!!");
            }

        }
        dataLoader.AllocData(startIndices, mSolverData, dev_Edges, dev_TetFathers);
        for (auto softBodyData : dataLoader.m_softBodyData) {
            softBodies.push_back(new SoftBody(this, std::get<2>(softBodyData), &std::get<1>(softBodyData)));
        }
        mSolverParams.pCollisionDetection->Init(mSolverData.numTets, mSolverData.numVerts, maxThreads);
        cudaMalloc((void**)&mSolverData.dev_Normals, mSolverData.numVerts * sizeof(glm::vec3));
        cudaMalloc((void**)&mSolverData.dev_tIs, mSolverData.numVerts * sizeof(dataType));
    }
    mSolverData.pFixedBodies = new FixedBodyData{ _threadsPerBlock, _fixedBodies };
    mSolver = new IPCSolver{ threadsPerBlock, mSolverData };
}

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
    cudaFree(mSolverData.XTilde);
    for (auto softbody : softBodies) {
        delete softbody;
    }
    cudaFree(mSolverData.dev_Normals);
    delete mSolver;
}

int SimulationCUDAContext::GetVertCnt() const {
    return mSolverData.numVerts;
}

int SimulationCUDAContext::GetNumQueries() const {
    return mSolverParams.pCollisionDetection->GetNumQueries();
}

int SimulationCUDAContext::GetTetCnt() const {
    return mSolverData.numTets;
}

template<typename HighP>
void DataLoader<HighP>::CollectData(const char* nodeFileName, const char* eleFileName, const char* faceFileName, const glm::vec3& pos, const glm::vec3& scale, const glm::vec3& rot,
    bool centralize, int startIndex, SolverAttribute attrib)
{
    SolverData<HighP> solverData;
    SoftBodyData softBodyData;
    auto vertices = loadNodeFile(nodeFileName, centralize, solverData.numVerts);
    cudaMalloc((void**)&solverData.X, sizeof(glm::tvec3<HighP>) * solverData.numVerts);
    cudaMemcpy(solverData.X, vertices.data(), sizeof(glm::tvec3<HighP>) * solverData.numVerts, cudaMemcpyHostToDevice);

    // transform
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, pos);
    model = glm::scale(model, scale);
    model = glm::rotate(model, glm::radians(rot.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
    int blocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    TransformVertices << < blocks, threadsPerBlock >> > (solverData.X, model, solverData.numVerts);

    auto tetIdx = loadEleFile(eleFileName, startIndex, solverData.numTets);
    cudaMalloc((void**)&solverData.Tet, sizeof(indexType) * tetIdx.size());
    cudaMemcpy(solverData.Tet, tetIdx.data(), sizeof(indexType) * tetIdx.size(), cudaMemcpyHostToDevice);
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
    totalNumVerts += solverData.numVerts;
    totalNumTets += solverData.numTets;

    m_softBodyData.push_back({ solverData,softBodyData, attrib });
}

template<typename HighP>
void DataLoader<HighP>::CollectData(const char* mshFileName, const glm::vec3& pos, const glm::vec3& scale, const glm::vec3& rot,
    bool centralize, int startIndex, SolverAttribute attrib)
{
    SolverData<HighP> solverData;
    SoftBodyData softBodyData;
    igl::MshLoader _loader(mshFileName);
    auto nodes = _loader.get_nodes();
    std::vector<float> vertices(nodes.size());
    solverData.numVerts = nodes.size() / 3;
    std::transform(nodes.begin(), nodes.end(), vertices.begin(), [](igl::MshLoader::Float f) {
        return static_cast<float>(f);
        });
    cudaMalloc((void**)&solverData.X, sizeof(glm::tvec3<HighP>) * solverData.numVerts);
    cudaMemcpy(solverData.X, vertices.data(), sizeof(glm::tvec3<HighP>) * solverData.numVerts, cudaMemcpyHostToDevice);

    // transform
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, pos);
    model = glm::scale(model, scale);
    model = glm::rotate(model, glm::radians(rot.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
    int blocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    TransformVertices << < blocks, threadsPerBlock >> > (solverData.X, model, solverData.numVerts);

    auto elements = _loader.get_elements();
    std::vector<indexType> tetIdx(elements.size());
    std::transform(elements.begin(), elements.end(), tetIdx.begin(), [](int i) {
        return static_cast<indexType>(i);
        });
    solverData.numTets = tetIdx.size() / 4;
    cudaMalloc((void**)&solverData.Tet, sizeof(indexType) * tetIdx.size());
    cudaMemcpy(solverData.Tet, tetIdx.data(), sizeof(indexType) * tetIdx.size(), cudaMemcpyHostToDevice);
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
    totalNumVerts += solverData.numVerts;
    totalNumTets += solverData.numTets;

    m_softBodyData.push_back({ solverData,softBodyData, attrib });
}

template<typename HighP>
void DataLoader<HighP>::AllocData(std::vector<int>& startIndices, SolverData<HighP> &solverData, indexType*& gEdges, indexType*& gTetFather)
{
    solverData.numVerts = totalNumVerts;
    solverData.numTets = totalNumTets;
    cudaMalloc((void**)&solverData.X, sizeof(glm::tvec3<HighP>) * totalNumVerts);
    cudaMalloc((void**)&solverData.X0, sizeof(glm::tvec3<HighP>) * totalNumVerts);
    cudaMalloc((void**)&solverData.XTilde, sizeof(glm::tvec3<HighP>) * totalNumVerts);
    cudaMalloc((void**)&solverData.V, sizeof(glm::tvec3<HighP>) * totalNumVerts);
    cudaMalloc((void**)&solverData.dev_ExtForce, sizeof(glm::tvec3<HighP>) * totalNumVerts);
    cudaMemset(solverData.V, 0, sizeof(glm::tvec3<HighP>) * totalNumVerts);
    cudaMemset(solverData.dev_ExtForce, 0, sizeof(glm::tvec3<HighP>) * totalNumVerts);
    cudaMalloc((void**)&solverData.Tet, sizeof(indexType) * totalNumTets * 4);
    cudaMalloc((void**)&solverData.mass, sizeof(HighP) * totalNumVerts);
    cudaMalloc((void**)&gEdges, sizeof(indexType) * totalNumEdges * 2);
    cudaMalloc((void**)&gTetFather, sizeof(indexType) * totalNumTets);
    int vertOffset = 0, tetOffset = 0, edgeOffset = 0;
    for (int i = 0; i < m_softBodyData.size(); i++)
    {
        auto& softBody = m_softBodyData[i];
        startIndices.push_back(vertOffset);
        auto& softBodySolverData = std::get<0>(softBody);
        auto& softBodyData = std::get<1>(softBody);
        cudaMemcpy(solverData.X + vertOffset, softBodySolverData.X, sizeof(glm::tvec3<HighP>) * softBodySolverData.numVerts, cudaMemcpyDeviceToDevice);
        thrust::transform(softBodySolverData.Tet, softBodySolverData.Tet + softBodySolverData.numTets * 4, thrust::device_pointer_cast(solverData.Tet) + tetOffset, [vertOffset] __device__(indexType x) {
            return x + vertOffset;
        });
        if (softBodyData.Tri) {
            thrust::for_each(thrust::device_pointer_cast(softBodyData.Tri), thrust::device_pointer_cast(softBodyData.Tri) + softBodyData.numTris * 3, [vertOffset] __device__(indexType & x) {
                x += vertOffset;
            });
        }
        thrust::fill(thrust::device_pointer_cast(gTetFather) + tetOffset / 4, thrust::device_pointer_cast(gTetFather) + tetOffset / 4 + softBodySolverData.numTets, i);
        thrust::fill(thrust::device_pointer_cast(solverData.mass) + vertOffset, thrust::device_pointer_cast(solverData.mass) + vertOffset + softBodySolverData.numVerts, std::get<2>(softBody).mass);
        cudaMemcpy(gEdges + edgeOffset, m_edges[i].data(), sizeof(indexType) * m_edges[i].size(), cudaMemcpyHostToDevice);
        thrust::transform(thrust::device_pointer_cast(gEdges) + edgeOffset, thrust::device_pointer_cast(gEdges) + edgeOffset + m_edges[i].size(), thrust::device_pointer_cast(gEdges) + edgeOffset,
            [vertOffset] __device__(indexType x) {
            return x + vertOffset;
        });
        cudaFree(softBodySolverData.X);
        cudaFree(softBodySolverData.Tet);
        std::get<1>(softBody).numTets = softBodySolverData.numTets;
        std::get<1>(softBody).Tet = solverData.Tet + tetOffset;
        vertOffset += softBodySolverData.numVerts;
        tetOffset += softBodySolverData.numTets * 4;
        edgeOffset += m_edges[i].size();
    }
    cudaMemcpy(solverData.X0, solverData.X, sizeof(glm::tvec3<HighP>) * totalNumVerts, cudaMemcpyDeviceToDevice);
    cudaMemcpy(solverData.XTilde, solverData.X, sizeof(glm::tvec3<HighP>) * totalNumVerts, cudaMemcpyDeviceToDevice);
}

void SimulationCUDAContext::PrepareRenderData() {
    for (auto softbody : softBodies) {
        glm::vec3* pos;
        glm::vec4* nor;
        softbody->Mesh::MapDevicePtr(&pos, &nor);
        if (softbody->GetNumTris() == 0) {
            dim3 numThreadsPerBlock(softbody->GetNumTets() / threadsPerBlock + 1);
            PopulatePos << <numThreadsPerBlock, threadsPerBlock >> > (pos, mSolverData.X, softbody->GetSoftBodyData().Tet, softbody->GetNumTets());
            RecalculateNormals << <softbody->GetNumTets() * 4 / threadsPerBlock + 1, threadsPerBlock >> > (nor, pos, 4 * softbody->GetNumTets());
            softbody->Mesh::UnMapDevicePtr();
        }
        else {
            dim3 numThreadsPerBlock(softbody->GetNumTris() / threadsPerBlock + 1);
            PopulateTriPos << <numThreadsPerBlock, threadsPerBlock >> > (pos, mSolverData.X, softbody->GetSoftBodyData().Tri, softbody->GetNumTris());
            RecalculateNormals << <softbody->GetNumTris() / threadsPerBlock + 1, threadsPerBlock >> > (nor, pos, softbody->GetNumTris());
            softbody->Mesh::UnMapDevicePtr();
        }
    }
}