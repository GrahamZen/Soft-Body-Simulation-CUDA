#pragma once

#include <def.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <set>

template<typename HighP>
class DataLoader {
    friend class SimulationCUDAContext;
public:
    DataLoader(const int _threadsPerBlock) :threadsPerBlock(_threadsPerBlock) {}
    void CollectData(const char* nodeFileName, const char* eleFileName, const char* faceFileName, const glm::vec3& pos, const glm::vec3& scale,
        const glm::vec3& rot, bool centralize, int startIndex, SoftBodyAttribute attrib);
    void CollectData(const char* mshFileName, const glm::vec3& pos, const glm::vec3& scale, const glm::vec3& rot,
        bool centralize, int startIndex, SoftBodyAttribute attrib);
    void AllocData(std::vector<int>& startIndices, SolverData<HighP> &solverData, indexType*& edges, indexType*& tetFather);
private:
    static std::vector<indexType> loadEleFile(const std::string& EleFilename, int startIndex, int& numTets);
    static std::vector<glm::tvec3<HighP>> loadNodeFile(const std::string& nodeFilename, bool centralize, int& numVerts);
    static std::vector<indexType> loadFaceFile(const std::string& faceFilename, int startIndex, int& numTris);
    void CollectEdges(const std::vector<indexType>& triIdx);
    std::vector<std::tuple<SolverData<HighP>, SoftBodyData, SoftBodyAttribute>> m_softBodyData;
    std::vector<std::vector<indexType>> m_edges;
    int totalNumDBC = 0;
    int totalNumVerts = 0;
    int totalNumTets = 0;
    int totalNumEdges = 0;
    const int threadsPerBlock;
};

namespace fs = std::filesystem;

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
std::vector<glm::tvec3<HighP>> DataLoader<HighP>::loadNodeFile(const std::string& nodeFilename, bool centralize, int& numVerts)
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
    std::vector<glm::tvec3<HighP>> X(numVerts);
    glm::tvec3<HighP> center(0.0f);

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