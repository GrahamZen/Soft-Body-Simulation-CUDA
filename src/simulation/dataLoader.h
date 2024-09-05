#pragma once

#include <def.h>
#include <vector>
#include <string>
#include <memory>

class SoftBodyAttribute;
class SoftBody;
template<typename Scalar>
class DataLoader {
    friend class SimulationCUDAContext;
    struct Impl;
public:
    DataLoader(int _threadsPerBlock);
    ~DataLoader();
    void CollectData(const char* nodeFileName, const char* eleFileName, const char* faceFileName, const glm::vec3& pos, const glm::vec3& scale,
        const glm::vec3& rot, bool centralize, int startIndex, SoftBodyAttribute* attrib);
    void CollectData(const char* mshFileName, const glm::vec3& pos, const glm::vec3& scale, const glm::vec3& rot,
        bool centralize, int startIndex, SoftBodyAttribute* attrib);
    void AllocData(std::vector<int>& startIndices, SolverData<Scalar>& solverData, std::vector<SoftBody*>& softbodies);
private:
    static std::pair<std::vector<indexType>, std::vector<indexType>> loadEleFaceFile(const std::string& EleFilename, int startIndex, int& numTets, int& numTris, std::string faceFilename = "");
    static std::vector<glm::tvec3<Scalar>> loadNodeFile(const std::string& nodeFilename, bool centralize, int& numVerts);
    void CollectEdges(const std::vector<indexType>& triIdx);
    std::unique_ptr<Impl> m_impl;
    int totalNumDBC = 0;
    int totalNumVerts = 0;
    int totalNumTris = 0;
    int totalNumTets = 0;
    int totalNumEdges = 0;
    const int threadsPerBlock;
};