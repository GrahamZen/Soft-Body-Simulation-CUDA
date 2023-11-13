#include <softBody.h>
#include <shaderprogram.h>

class SimulationCUDAContext {
public:
    SimulationCUDAContext();
    ~SimulationCUDAContext();
    void Update();
    void Reset();
    float getDt() { return dt; }
    void setSoftBodyAttrJump(int index, bool jump) { softBodies[index]->setJump(jump); }
    void setDt(float dt) { this->dt = dt; }
    void addSoftBody(SoftBody*);
    void draw(ShaderProgram*);
private:
    std::vector<SoftBody*> softBodies;
    float dt = 0.001f;
};