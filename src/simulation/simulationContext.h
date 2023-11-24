#include <softBody.h>
#include <shaderprogram.h>
#include <json.hpp>

struct SoftBodyData {
    GLuint* Tets;
    glm::vec3* dev_X;
    int numTets;
    int numVerts;
};

class DataLoader {
    friend class SimulationCUDAContext;
public:
    DataLoader() = default;
    void CollectData(const char* nodeFileName, const char* eleFileName, const glm::vec3& pos, const glm::vec3& scale,
        const glm::vec3& rot, bool centralize, int startIndex, SoftBody::SoftBodyAttribute attrib);
private:
    glm::vec3* AllocData(std::vector<int>& startIndices);
    static std::vector<GLuint> loadEleFile(const std::string& EleFilename, int startIndex, int& numTets);
    static std::vector<glm::vec3> loadNodeFile(const std::string& nodeFilename, bool centralize, int& numVerts);
    std::vector<std::pair<SoftBodyData, SoftBody::SoftBodyAttribute>> m_softBodyData;
    glm::vec3* dev_XPtr;
    int totalNumVerts = 0;
};

class SimulationCUDAContext {
public:
    SimulationCUDAContext(Context* ctx, nlohmann::json& json);
    ~SimulationCUDAContext();
    void Update();
    void Reset();
    float GetDt() { return dt; }
    float GetGravity() { return gravity; }
    void UpdateSingleSBAttr(int index, GuiDataContainer::SoftBodyAttr& softBodyAttr);
    void SetDt(float dt) { this->dt = dt; }
    void Draw(ShaderProgram*);
    AABB GetAABB() const;
    const BVH* GetBVHPtr() const { return &m_bvh; };
    int GetTetCnt() const;
    int GetVertCnt() const;
    void CCD();
private:
    glm::vec3* dev_Xs;
    std::vector<SoftBody*> softBodies;
    std::vector<int> startIndices;
    BVH m_bvh;
    float dt = 0.001f;
    float gravity = 9.8f;
    Context* context = nullptr;
};