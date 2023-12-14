#pragma once

#include <simulation/solver/solver.h>
class FEMSolver : public Solver {
public:
    FEMSolver(SimulationCUDAContext*, SolverAttribute&, SolverData*);
private:

    indexType* Tet;

    int numTets; // The number of tetrahedra
};
