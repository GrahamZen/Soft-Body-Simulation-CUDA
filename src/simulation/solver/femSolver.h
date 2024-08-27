#pragma once

#include <simulation/solver/solver.h>

template<typename HighP>
class FEMSolver : public Solver<HighP> {
public:
    FEMSolver(int threadsPerBlock, const SolverData<HighP>& solverData);
    virtual ~FEMSolver() = default;

    virtual void Update(SolverData<HighP>& solverData, SolverParams& solverParams) = 0;
protected:
    virtual void SolverPrepare(SolverData<HighP>& solverData, SolverParams& solverParams) = 0;
    virtual void SolverStep(SolverData<HighP>& solverData, SolverParams& solverParams) = 0;
};

template<typename HighP>
FEMSolver<HighP>::FEMSolver(int threadsPerBlock, const SolverData<HighP>& solverData) : Solver<HighP>(threadsPerBlock) {
    cudaMalloc((void**)&solverData.V0, sizeof(HighP) * solverData.numTets);
    cudaMemset(solverData.V0, 0, sizeof(HighP) * solverData.numTets);
    cudaMalloc((void**)&solverData.DmInv, sizeof(glm::tmat4x4<HighP>) * solverData.numTets);

    int blocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    computeInvDmV0 << < blocks, threadsPerBlock >> > (solverData.V0, solverData.DmInv, solverData.numTets, solverData.X, solverData.Tet);
}