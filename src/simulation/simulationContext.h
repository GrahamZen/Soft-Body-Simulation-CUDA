#pragma once

#include <collision/bvh.h>
#include <fixedBodyData.h>
#include <simulation/solver/solver.h>
#include <context.h>
#include <json.hpp>

class SoftBody;
class SurfaceShader;
class DataLoader {
    friend class SimulationCUDAContext;
public:
    DataLoader(const int);
    void CollectData(const char* nodeFileName, const char* eleFileName, const char* faceFileName, const glm::vec3& pos, const glm::vec3& scale,
        const glm::vec3& rot, bool centralize, int startIndex, SolverAttribute attrib);
    void CollectData(const char* mshFileName, const glm::vec3& pos, const glm::vec3& scale, const glm::vec3& rot,
        bool centralize, int startIndex, SolverAttribute attrib);
    void AllocData(std::vector<int>& startIndices, glm::vec3*& gX, glm::vec3*& gX0, glm::vec3*& gXTilt, glm::vec3*& gV, glm::vec3*& gF, indexType*& gEdges, indexType*& gTet, indexType*& gTetFather, int& numVerts, int& numTets);
private:
    static std::vector<indexType> loadEleFile(const std::string& EleFilename, int startIndex, int& numTets);
    static std::vector<glm::vec3> loadNodeFile(const std::string& nodeFilename, bool centralize, int& numVerts);
    static std::vector<indexType> loadFaceFile(const std::string& faceFilename, int startIndex, int& numTris);
    void CollectEdges(const std::vector<indexType>& triIdx);
    std::vector<std::tuple<SolverData, SoftBodyData, SolverAttribute>> m_softBodyData;
    std::vector<std::vector<indexType>> m_edges;
    int totalNumVerts = 0;
    int totalNumTets = 0;
    int totalNumEdges = 0;
    const int threadsPerBlock;
};

class SimulationCUDAContext {
    friend class CollisionDetection;
public:

    SimulationCUDAContext(Context* ctx, const std::string& _name, nlohmann::json& json,
        const std::map<std::string, nlohmann::json>& softBodyDefs, std::vector<FixedBody*>&, int threadsPerBlock, int _threadsPerBlockBVH, int _maxThreads, int _numIterations);
    ~SimulationCUDAContext();
    void Update();
    void Reset();
    const std::vector<const char*>& GetNamesSoftBodies() const { return namesSoftBodies; }
    void UpdateSingleSBAttr(int index, GuiDataContainer::SoftBodyAttr& softBodyAttr);
    void SetDt(float dt) { mSolverParams.dt = dt; }
    void SetBVHBuildType(BVH::BuildType);
    void SetGlobalSolver(bool useEigen);
    void Draw(SurfaceShader*, SurfaceShader*);
    const SolverParams& GetSolverParams() const;
    int GetTetCnt() const;
    int GetVertCnt() const;
    int GetThreadsPerBlock() const { return threadsPerBlock; }
    int GetNumQueries() const;
private:
    void CCD();
    void PrepareRenderData();
    int threadsPerBlock = 64;
    SolverData mSolverData;
    dataType* dev_tIs;
    glm::vec3* dev_Normals;
    indexType* dev_TetFathers;
    indexType* dev_Edges;
    std::vector<const char*> namesSoftBodies;
    std::vector<SoftBody*> softBodies;
    std::vector<FixedBody*> fixedBodies;
    FixedBodyData dev_fixedBodies;
    std::vector<int> startIndices;
    CollisionDetection mCollisionDetection;

    Context* context = nullptr;
    const std::string name;
    SolverParams mSolverParams;
    Solver* mSolver = nullptr;
};