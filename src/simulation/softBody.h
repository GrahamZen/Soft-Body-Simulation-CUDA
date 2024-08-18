#pragma once

#include <mesh.h>
#include <context.h>
#include <simulation/solver/femSolver.h>
#include <vector>

class SimulationCUDAContext;
class SolverData;

class SoftBody : public Mesh {
public:
    SoftBody(SimulationCUDAContext*, SolverAttribute&, SoftBodyData*);
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
    SolverAttribute attrib;
    bool jump = false;
    const int threadsPerBlock;
};
