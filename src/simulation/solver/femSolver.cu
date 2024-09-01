#include <simulation/solver/femsolver.h>
#include <utilities.cuh>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <solver/solverUtil.cuh>

template<typename HighP>
FEMSolver<HighP>::FEMSolver(int threadsPerBlock, const SolverData<HighP>& solverData) : Solver<HighP>(threadsPerBlock) {
    cudaMalloc((void**)&solverData.V0, sizeof(HighP) * solverData.numTets);
    cudaMemset(solverData.V0, 0, sizeof(HighP) * solverData.numTets);
    cudaMalloc((void**)&solverData.DmInv, sizeof(glm::tmat4x4<HighP>) * solverData.numTets);

    int blocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    thrust::device_vector<HighP> degree(solverData.numVerts);
    thrust::device_ptr<HighP> ptr(solverData.contact_area);
    computeInvDmV0 << < blocks, threadsPerBlock >> > (solverData.V0, solverData.DmInv, solverData.numTets, solverData.X, solverData.Tet, solverData.contact_area, degree.data().get());
    thrust::transform(ptr, ptr + solverData.numTets, degree.begin(), ptr, thrust::divides<HighP>());
}

template class FEMSolver<float>;
template class FEMSolver<double>;