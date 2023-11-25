#include <json.hpp>
#include <softBody.h>
#include <shaderprogram.h>
#include <bvh.h>
#include <sceneStructs.h>

struct SoftBodyData {
    GLuint* Tets;
    glm::vec3* dev_X;
    glm::vec3* dev_X0;
    glm::vec3* dev_XTilt;
    glm::vec3* dev_V;
    glm::vec3* dev_F;
    int numTets;
    int numVerts;
};

class DataLoader {
    friend class SimulationCUDAContext;
public:
    DataLoader(int&);
    void CollectData(const char* nodeFileName, const char* eleFileName, const glm::vec3& pos, const glm::vec3& scale,
        const glm::vec3& rot, bool centralize, int startIndex, SoftBody::SoftBodyAttribute attrib);
    void AllocData(std::vector<int>& startIndices, glm::vec3*& gX, glm::vec3*& gX0, glm::vec3*& gXTilt, glm::vec3*& gV, glm::vec3*& gF, GLuint*& gTet, int& numVerts, int& numTets);
private:
    static std::vector<GLuint> loadEleFile(const std::string& EleFilename, int startIndex, int& numTets);
    static std::vector<glm::vec3> loadNodeFile(const std::string& nodeFilename, bool centralize, int& numVerts);
    std::vector<std::pair<SoftBodyData, SoftBody::SoftBodyAttribute>> m_softBodyData;
    int totalNumVerts = 0;
    int totalNumTets = 0;
    int& threadsPerBlock;
};

class SimulationCUDAContext {
public:
    SimulationCUDAContext(Context* ctx, nlohmann::json& json);
    ~SimulationCUDAContext();
    void Update();
    void Reset();
    const std::vector<const char*>& GetNamesSoftBodies() const { return namesSoftBodies; }
    float GetDt() const { return dt; }
    float GetGravity() const { return gravity; }
    void UpdateSingleSBAttr(int index, GuiDataContainer::SoftBodyAttr& softBodyAttr);
    void SetDt(float dt) { this->dt = dt; }
    void Draw(ShaderProgram*);
    AABB GetAABB() const;
    int GetTetCnt() const;
    int GetVertCnt() const;
    int& GetThreadsPerBlock() { return threadsPerBlock; }
    void CCD();
private:
    void PrepareRenderData();
    std::vector<const char*> namesSoftBodies;
    glm::vec3* dev_Xs;
    glm::vec3* dev_X0s;
    glm::vec3* dev_XTilts;
    glm::vec3* dev_Vs;
    glm::vec3* dev_Fs;
    GLuint* dev_Tets;
    GLuint* dev_TetIds;
    int numVerts = 0;
    int numTets = 0;
    std::vector<SoftBody*> softBodies;
    std::vector<int> startIndices;
    BVH m_bvh;
    float damp = 0.999f;
    float muN = 0.5f;
    float muT = 0.5f;
    float dt = 0.001f;
    float gravity = 9.8f;
    Context* context = nullptr;
    int threadsPerBlock = 64;
};