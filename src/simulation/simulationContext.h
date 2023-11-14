#include <softBody.h>
#include <shaderprogram.h>

class SimulationCUDAContext {
public:
    SimulationCUDAContext();
    ~SimulationCUDAContext();
    void Update();
    void Reset();
    float GetDt() { return dt; }
    void UpdateSingleSBAttr(int index, GuiDataContainer::SoftBodyAttr& softBodyAttr);
    void SetDt(float dt) { this->dt = dt; }
    void AddSoftBody(SoftBody*);
    void Draw(ShaderProgram*);
private:
    std::vector<SoftBody*> softBodies;
    float dt = 0.001f;
};