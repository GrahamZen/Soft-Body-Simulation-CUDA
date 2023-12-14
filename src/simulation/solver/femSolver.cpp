#pragma once

#include <solver.cuh>
class FEMSolver : public Solver {
public:
    FEMSolver(SimulationCUDAContext*, SolverAttribute&, SolverData*);
private:

    indexType* Tet;

    int numTets; // The number of tetrahedra
};
