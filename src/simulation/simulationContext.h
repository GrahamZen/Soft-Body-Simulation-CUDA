#include <softBody.h>

class SimulationCUDAContext {
public:
    SimulationCUDAContext();
    ~SimulationCUDAContext();
    void Update();
    std::vector<SoftBody*> softBodies;
};



void InitDataContainer(GuiDataContainer* guiData);