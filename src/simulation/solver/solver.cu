#include <simulation/solver/femSolver.h>
#include <simulation/simulationContext.h>

Solver::Solver(SimulationCUDAContext* context) :mcrpSimContext(context), threadsPerBlock(context->GetThreadsPerBlock())
{

}

Solver::~Solver() {
}

FEMSolver::FEMSolver(SimulationCUDAContext* context) : Solver(context) {}

