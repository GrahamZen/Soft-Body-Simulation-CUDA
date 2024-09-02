#pragma once

#include <mesh.h>
#include <context.h>
#include <simulation/solver/femSolver.h>
#include <vector>

class SimulationCUDAContext;

class SoftBody : public Mesh {
public:
    SoftBody(const SoftBodyData* dataPtr, const SoftBodyAttribute _attrib, int threadsPerBlock = 256);
    ~SoftBody();
    SoftBody(const SoftBody&) = delete;
    SoftBody& operator=(const SoftBody&) = delete;
    int GetNumTets()const;
    int GetNumTris()const;
    void Reset();
    const SoftBodyData& GetSoftBodyData() const;
    void SetAttributes(GuiDataContainer::SoftBodyAttr& softBodyAttr);
private:
    SoftBodyData softBodyData;
    SoftBodyAttribute attrib;
    bool jump = false;
    const int threadsPerBlock;
};
