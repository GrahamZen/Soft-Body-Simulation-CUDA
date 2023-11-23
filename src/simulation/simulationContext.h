#include <softBody.h>
#include <shaderprogram.h>
#include <json.hpp>

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
    std::vector<SoftBody*> softBodies;
    BVH m_bvh;
    float dt = 0.001f;
    float gravity = 9.8f;
    Context* context = nullptr;
};