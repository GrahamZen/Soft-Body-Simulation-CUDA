#include <simulation/solver/femSolver.h>
#include <simulation/simulationContext.h>

template<typename HighP>
Solver<HighP>::Solver(int threadsPerBlock) : threadsPerBlock(threadsPerBlock)
{
}

template<typename HighP>
Solver<HighP>::~Solver() {
}