#pragma once

#include <vector>
#include <mesh.h>
#include <context.h>
#include <simulation/solver/femSolver.h>

class SimulationCUDAContext;
class SolverData;

class SoftBody : public Mesh {
public:
    SoftBody(SimulationCUDAContext*, SolverAttribute&, SolverData*);
    ~SoftBody();
    SoftBody(const SoftBody&) = delete;
    SoftBody& operator=(const SoftBody&) = delete;
    int GetNumVerts()const;
    int GetNumTets()const;
    int GetNumTris()const;
    void Update();
    void Reset();
    const SolverData& GetSolverData() const;
    void SetAttributes(GuiDataContainer::SoftBodyAttr& softBodyAttr);
private:
    SolverData solverData;
    SolverAttribute attrib;
    FEMSolver* solver;
    bool jump = false;
    glm::vec3* X0;
    const int threadsPerBlock;
};
