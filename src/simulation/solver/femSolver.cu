#include <simulation/solver/femSolver.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <solver/solverUtil.cuh>

template<typename Scalar>
FEMSolver<Scalar>::FEMSolver(int threadsPerBlock, const SolverData<Scalar>& solverData) : Solver<Scalar>(threadsPerBlock) {
    cudaMalloc((void**)&solverData.V0, sizeof(Scalar) * solverData.numTets);
    cudaMemset(solverData.V0, 0, sizeof(Scalar) * solverData.numTets);
    cudaMalloc((void**)&solverData.DmInv, sizeof(glm::tmat4x4<Scalar>) * solverData.numTets);

    int blocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    thrust::device_vector<Scalar> degree(solverData.numVerts);
    thrust::device_ptr<Scalar> ptr(solverData.contact_area);
    computeInvDmV0 << < blocks, threadsPerBlock >> > (solverData.V0, solverData.DmInv, solverData.numTets, solverData.X, solverData.Tet, solverData.contact_area, degree.data().get());
    thrust::transform(ptr, ptr + solverData.numVerts, degree.begin(), ptr, thrust::divides<Scalar>());
}

template class FEMSolver<float>;
template class FEMSolver<double>;