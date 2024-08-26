#pragma once

#include <simulation/solver/femSolver.h>


template<typename HighP>
FEMSolver<HighP>::FEMSolver(int threadsPerBlock) : Solver(threadsPerBlock) {
}