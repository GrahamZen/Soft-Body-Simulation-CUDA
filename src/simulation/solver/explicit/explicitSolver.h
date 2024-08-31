#pragma once
#include <simulation/solver/femSolver.h>

class ExplicitSolver : public FEMSolver<float> {
public:
    ExplicitSolver(int threadsPerBlock, const SolverData<float>& solverData);
    ~ExplicitSolver();
    virtual void Update(SolverData<float>& solverData, SolverParams& solverParams) override;
protected:
    virtual void SolverPrepare(SolverData<float>& solverData, SolverParams& solverParams) override;
    virtual void SolverStep(SolverData<float>& solverData, SolverParams& solverParams) override;
private:
    void Laplacian_Smoothing(SolverData<float>& solverData, float blendAlpha = 0.5f);
    // For Laplacian smoothing.
    glm::vec3* V_sum;
    int* V_num;
};
