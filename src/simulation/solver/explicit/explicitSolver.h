#pragma once
#include <simulation/solver/femSolver.h>
#include <context.h>

class SimulationCUDAContext;
class ExplicitSolver : public FEMSolver {
public:
    ExplicitSolver(SimulationCUDAContext*, const SolverData&);
    ~ExplicitSolver();
    virtual void Update(SolverData& solverData, SolverAttribute& solverAttr) override;
protected:
    virtual void SolverPrepare(SolverData& solverData, SolverAttribute& solverAttr) override;
    virtual void SolverStep(SolverData& solverData, SolverAttribute& solverAttr) override;
private:
    void Laplacian_Smoothing(SolverData& solverData, float blendAlpha = 0.5f);
    // For Laplacian smoothing.
    glm::vec3* V_sum;
    int* V_num;
};
