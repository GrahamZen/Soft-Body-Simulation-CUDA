#include <softBody.h>
#include <shaderprogram.h>

class SimulationCUDAContext {
public:
    SimulationCUDAContext();
    ~SimulationCUDAContext();
    void Update();
    void Reset();
    float GetDt() { return dt; }
    float GetGravity() { return gravity; }
    void UpdateSingleSBAttr(int index, GuiDataContainer::SoftBodyAttr& softBodyAttr);
    void SetDt(float dt) { this->dt = dt; }
    void SetGravity(float g) { gravity = g; }
    void AddSoftBody(SoftBody*);
    void Draw(ShaderProgram*);
private:
    std::vector<SoftBody*> softBodies;
    float dt = 0.001f;
    float gravity = 9.8f;
};