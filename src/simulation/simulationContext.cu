#include <utilities.cuh>
#include <simulation/solver/projective/pdSolver.h>
#include <simulation/softBody.h>
#include <simulation/MshLoader.h>
#include <simulation/simulationContext.h>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <set>

namespace fs = std::filesystem;

template<typename HighP>
DataLoader<HighP>::DataLoader(const int _threadsPerBlock) :threadsPerBlock(_threadsPerBlock)
{
}

template<typename HighP>
std::vector<indexType> DataLoader<HighP>::loadEleFile(const std::string& EleFilename, int startIndex, int& numTets)
{
    std::string line;
    std::ifstream file(EleFilename);

    if (!file.is_open()) {
        fs::path absolutePath = fs::absolute(EleFilename);
        std::cerr << "Unable to open file: " << absolutePath << std::endl;
    }

    std::getline(file, line);
    std::istringstream iss(line);
    iss >> numTets;

    std::vector<indexType> Tet(numTets * 4);

    int a, b, c, d, e;
    for (int tet = 0; tet < numTets && std::getline(file, line); ++tet) {
        std::istringstream iss(line);
        iss >> a >> b >> c >> d >> e;

        Tet[tet * 4 + 0] = b - startIndex;
        Tet[tet * 4 + 1] = c - startIndex;
        Tet[tet * 4 + 2] = d - startIndex;
        Tet[tet * 4 + 3] = e - startIndex;
    }

    file.close();
    return Tet;
}

template<typename HighP>
std::vector<indexType> DataLoader<HighP>::loadFaceFile(const std::string& faceFilename, int startIndex, int& numTris)
{
    std::string line;
    std::ifstream file(faceFilename);

    if (!file.is_open()) {
        // std::cerr << "Unable to open face file" << std::endl;
        return std::vector<indexType>();
    }

    std::getline(file, line);
    std::istringstream iss(line);
    iss >> numTris;

    std::vector<indexType> Triangle(numTris * 3);

    int a, b, c, d, e;
    for (int tet = 0; tet < numTris && std::getline(file, line); ++tet) {
        std::istringstream iss(line);
        iss >> a >> b >> c >> d >> e;

        Triangle[tet * 3 + 0] = b - startIndex;
        Triangle[tet * 3 + 1] = c - startIndex;
        Triangle[tet * 3 + 2] = d - startIndex;
    }

    file.close();
    return Triangle;
}

template<typename HighP>
std::vector<glm::vec3> DataLoader<HighP>::loadNodeFile(const std::string& nodeFilename, bool centralize, int& numVerts)
{
    std::ifstream file(nodeFilename);
    if (!file.is_open()) {
        fs::path absolutePath = fs::absolute(nodeFilename);
        std::cerr << "Unable to open file: " << absolutePath << std::endl;
        return {};
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> numVerts;
    std::vector<glm::vec3> X(numVerts);
    glm::vec3 center(0.0f);

    for (int i = 0; i < numVerts && std::getline(file, line); ++i) {
        std::istringstream lineStream(line);
        int index;
        float x, y, z;
        lineStream >> index >> x >> y >> z;

        X[i].x = x;
        X[i].y = y;
        X[i].z = z;

        center += X[i];
    }

    // Centralize the model
    if (centralize) {
        center /= static_cast<float>(numVerts);
        for (int i = 0; i < numVerts; ++i) {
            X[i] -= center;
            float temp = X[i].y;
            X[i].y = X[i].z;
            X[i].z = temp;
        }
    }

    return X;
}

template<typename HighP>
void DataLoader<HighP>::CollectEdges(const std::vector<indexType>& triIdx) {
    std::set<std::pair<indexType, indexType>> uniqueEdges;
    std::vector<indexType> edges;

    for (size_t i = 0; i < triIdx.size(); i += 3) {
        indexType v0 = triIdx[i];
        indexType v1 = triIdx[i + 1];
        indexType v2 = triIdx[i + 2];

        std::pair<indexType, indexType> edge1 = std::minmax(v0, v1);
        std::pair<indexType, indexType> edge2 = std::minmax(v1, v2);
        std::pair<indexType, indexType> edge3 = std::minmax(v2, v0);

        uniqueEdges.insert(edge1);
        uniqueEdges.insert(edge2);
        uniqueEdges.insert(edge3);
    }

    for (const auto& edge : uniqueEdges) {
        edges.push_back(edge.first);
        edges.push_back(edge.second);
    }

    m_edges.push_back(edges);
    totalNumEdges += edges.size() / 2;
}
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
    DataLoader<float> dataLoader(threadsPerBlock);
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
        dataLoader.AllocData(startIndices, mSolverData.X, mSolverData.X0, mSolverData.XTilde, mSolverData.V, mSolverData.Force, dev_Edges, mSolverData.Tet, dev_TetFathers, mSolverData.numVerts, mSolverData.numTets);
        for (auto softBodyData : dataLoader.m_softBodyData) {
            softBodies.push_back(new SoftBody(this, std::get<2>(softBodyData), &std::get<1>(softBodyData)));
        }
        mSolverParams.pCollisionDetection->Init(mSolverData.numTets, mSolverData.numVerts, maxThreads);
        cudaMalloc((void**)&mSolverData.dev_Normals, mSolverData.numVerts * sizeof(glm::vec3));
        cudaMalloc((void**)&mSolverData.dev_tIs, mSolverData.numVerts * sizeof(dataType));
    }
    mSolverData.pFixedBodies = new FixedBodyData{ _threadsPerBlock, _fixedBodies };
    mSolver = new PdSolver{ threadsPerBlock, mSolverData };
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
    cudaMalloc((void**)&solverData.X, sizeof(glm::vec3) * solverData.numVerts);
    cudaMemcpy(solverData.X, vertices.data(), sizeof(glm::vec3) * solverData.numVerts, cudaMemcpyHostToDevice);

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
    cudaMalloc((void**)&solverData.X, sizeof(glm::vec3) * solverData.numVerts);
    cudaMemcpy(solverData.X, vertices.data(), sizeof(glm::vec3) * solverData.numVerts, cudaMemcpyHostToDevice);

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
void DataLoader<HighP>::AllocData(std::vector<int>& startIndices, glm::vec3*& gX, glm::vec3*& gX0, glm::vec3*& gXTilde,
    glm::vec3*& gV, glm::vec3*& gF, indexType*& gEdges, indexType*& gTet, indexType*& gTetFather, int& numVerts, int& numTets)
{
    numVerts = totalNumVerts;
    numTets = totalNumTets;
    cudaMalloc((void**)&gX, sizeof(glm::vec3) * totalNumVerts);
    cudaMalloc((void**)&gX0, sizeof(glm::vec3) * totalNumVerts);
    cudaMalloc((void**)&gXTilde, sizeof(glm::vec3) * totalNumVerts);
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
        auto& softBody = m_softBodyData[i];
        startIndices.push_back(vertOffset);
        auto& solverData = std::get<0>(softBody);
        auto& softBodyData = std::get<1>(softBody);
        cudaMemcpy(gX + vertOffset, solverData.X, sizeof(glm::vec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
        thrust::transform(solverData.Tet, solverData.Tet + solverData.numTets * 4, dev_gTetPtr + tetOffset, [vertOffset] __device__(indexType x) {
            return x + vertOffset;
        });
        if (softBodyData.Tri) {
            thrust::for_each(thrust::device_pointer_cast(softBodyData.Tri), thrust::device_pointer_cast(softBodyData.Tri) + softBodyData.numTris * 3, [vertOffset] __device__(indexType & x) {
                x += vertOffset;
            });
        }
        thrust::fill(dev_gTetFatherPtr + tetOffset / 4, dev_gTetFatherPtr + tetOffset / 4 + solverData.numTets, i);
        cudaMemcpy(gEdges + edgeOffset, m_edges[i].data(), sizeof(indexType) * m_edges[i].size(), cudaMemcpyHostToDevice);
        thrust::transform(dev_gEdgesPtr + edgeOffset, dev_gEdgesPtr + edgeOffset + m_edges[i].size(), dev_gEdgesPtr + edgeOffset,
            [vertOffset] __device__(indexType x) {
            return x + vertOffset;
        });
        cudaFree(solverData.X);
        cudaFree(solverData.Tet);
        std::get<1>(softBody).numTets = solverData.numTets;
        std::get<1>(softBody).Tet = gTet + tetOffset;
        vertOffset += solverData.numVerts;
        tetOffset += solverData.numTets * 4;
        edgeOffset += m_edges[i].size();
    }
    cudaMemcpy(gX0, gX, sizeof(glm::vec3) * totalNumVerts, cudaMemcpyDeviceToDevice);
    cudaMemcpy(gXTilde, gX, sizeof(glm::vec3) * totalNumVerts, cudaMemcpyDeviceToDevice);
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