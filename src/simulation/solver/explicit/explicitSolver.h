#pragma once
#include <simulation/solver/femSolver.h>
#include <context.h>

class SimulationCUDAContext;
class ExplicitSolver : public FEMSolver {
public:
    ExplicitSolver(const SolverData&, int threadsPerBlock);
    ~ExplicitSolver();
    virtual void Update(SolverData& solverData, SolverParams& solverParams) override;
protected:
    virtual void SolverPrepare(SolverData& solverData, SolverParams& solverParams) override;
    virtual void SolverStep(SolverData& solverData, SolverParams& solverParams) override;
private:
    void Laplacian_Smoothing(SolverData& solverData, float blendAlpha = 0.5f);
    // For Laplacian smoothing.
    glm::vec3* V_sum;
    int* V_num;
};
