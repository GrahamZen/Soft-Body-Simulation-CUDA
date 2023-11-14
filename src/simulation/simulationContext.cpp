#include <simulationContext.h>

SimulationCUDAContext::SimulationCUDAContext()
{
}

SimulationCUDAContext::~SimulationCUDAContext()
{
    for (auto softbody : softBodies) {
        delete softbody;
    }
}
void SimulationCUDAContext::UpdateSingleSBAttr(int index, GuiDataContainer::SoftBodyAttr& softBodyAttr) {
    softBodies[index]->setAttributes(softBodyAttr);
}

void SimulationCUDAContext::Reset()
{
    for (auto softbody : softBodies) {
        softbody->Reset();
    }
}

void SimulationCUDAContext::AddSoftBody(SoftBody* softbody)
{
    softBodies.push_back(softbody);
}

void SimulationCUDAContext::Draw(ShaderProgram* shaderProgram)
{
    for (auto softBody : softBodies)
        shaderProgram->draw(*softBody, 0);
}
