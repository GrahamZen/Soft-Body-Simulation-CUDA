#include <json.hpp>
#include <softBody.h>
#include <surfaceshader.h>
#include <bvh.h>
#include <sceneStructs.h>
#include <fixedBodyData.h>

struct SoftBodyData {
    GLuint* Tets;
    GLuint* Tris;
    glm::vec3* dev_X;
    glm::vec3* dev_X0;
    glm::vec3* dev_XTilt;
    glm::vec3* dev_V;
    glm::vec3* dev_F;
    int numTets;
    int numVerts;
    int numTris;
};

class DataLoader {
    friend class SimulationCUDAContext;
public:
    DataLoader(const int);
    void CollectData(const char* nodeFileName, const char* eleFileName, const char* faceFileName, const glm::vec3& pos, const glm::vec3& scale,
        const glm::vec3& rot, bool centralize, int startIndex, SoftBody::SoftBodyAttribute attrib);
    void AllocData(std::vector<int>& startIndices, glm::vec3*& gX, glm::vec3*& gX0, glm::vec3*& gXTilt, glm::vec3*& gV, glm::vec3*& gF, GLuint*& gEdges, GLuint*& gTet, GLuint*& gTetFather, int& numVerts, int& numTets);
private:
    static std::vector<GLuint> loadEleFile(const std::string& EleFilename, int startIndex, int& numTets);
    static std::vector<glm::vec3> loadNodeFile(const std::string& nodeFilename, bool centralize, int& numVerts);
    static std::vector<GLuint> loadFaceFile(const std::string& faceFilename, int startIndex, int& numTris);
    void CollectEdges(const std::vector<GLuint>& triIdx);
    std::vector<std::pair<SoftBodyData, SoftBody::SoftBodyAttribute>> m_softBodyData;
    std::vector<std::vector<GLuint>> m_edges;
    int totalNumVerts = 0;
    int totalNumTets = 0;
    int totalNumEdges = 0;
    const int threadsPerBlock;
};


class SimulationCUDAContext {
    friend class CollisionDetection;
public:
    struct ExternalForce {
        glm::vec3 jump = glm::vec3(0.f, 400.f, 0.f);
    };
    SimulationCUDAContext(Context* ctx, const std::string& _name, const ExternalForce& extForce, nlohmann::json& json,
        const std::map<std::string, nlohmann::json>& softBodyDefs, std::vector<FixedBody*>&, int threadsPerBlock, int _threadsPerBlockBVH, int _maxThreads, int _numIterations);
    ~SimulationCUDAContext();
    void Update();
    void Reset();
    const std::vector<const char*>& GetNamesSoftBodies() const { return namesSoftBodies; }
    float GetDt() const { return dt; }
    float GetGravity() const { return gravity; }
    const ExternalForce& GetExtForce() const { return extForce; }
    void UpdateSingleSBAttr(int index, GuiDataContainer::SoftBodyAttr& softBodyAttr);
    void SetDt(float dt) { this->dt = dt; }
    void SetCUDASolver(bool useCUDASolver) { this->useCUDASolver = useCUDASolver; }
    void SetGlobalSolver(bool useEigen) { this->useEigen = useEigen; }
    bool IsEigenGlobalSolver() const { return useEigen; }
    bool IsCUDASolver() const { return useCUDASolver; }
    void Draw(SurfaceShader*, SurfaceShader*);
    AABB GetAABB() const;
    int GetTetCnt() const;
    int GetVertCnt() const;
    int GetThreadsPerBlock() const { return threadsPerBlock; }
    int GetNumQueries() const;
    int GetNumIterations() const { return numIterations; }
    void CCD();
private:
    ExternalForce extForce;
    float floorY = 0.f;
    glm::vec3 floorUp = glm::vec3(0.0f, 1.0f, 0.0f);
    void PrepareRenderData();
    int numIterations = 10;
    int threadsPerBlock = 64;
    bool useEigen = true;
    bool useCUDASolver = true;
    glm::vec3* dev_Xs;
    glm::vec3* dev_X0s;
    glm::vec3* dev_XTilts;
    glm::vec3* dev_Vs;
    glm::vec3* dev_Fs;
    dataType* dev_tIs;
    glm::vec3* dev_Normals;
    GLuint* dev_Tets;
    GLuint* dev_TetFathers;
    GLuint* dev_Edges;
    int numVerts = 0;
    int numTets = 0;
    std::vector<const char*> namesSoftBodies;
    std::vector<SoftBody*> softBodies;
    std::vector<FixedBody*> fixedBodies;
    FixedBodyData dev_fixedBodies;
    std::vector<int> startIndices;
    BVH m_bvh;
    float damp = 0.999f;
    float muN = 0.5f;
    float muT = 0.5f;
    float dt = 0.001f;
    float gravity = 9.8f;
    Context* context = nullptr;
    const std::string name;
};