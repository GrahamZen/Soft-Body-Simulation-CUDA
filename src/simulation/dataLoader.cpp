#include <simulation/simulationContext.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <set>

namespace fs = std::filesystem;

DataLoader::DataLoader(const int _threadsPerBlock) :threadsPerBlock(_threadsPerBlock)
{
}

std::vector<indexType> DataLoader::loadEleFile(const std::string& EleFilename, int startIndex, int& numTets)
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

std::vector<indexType> DataLoader::loadFaceFile(const std::string& faceFilename, int startIndex, int& numTris)
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

std::vector<glm::vec3> DataLoader::loadNodeFile(const std::string& nodeFilename, bool centralize, int& numVerts)
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

void DataLoader::CollectEdges(const std::vector<indexType>& triIdx) {
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