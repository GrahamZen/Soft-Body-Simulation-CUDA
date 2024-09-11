#include <energy/barrier.h>
#include <solver/solverUtil.cuh>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

namespace Barrier {
  
}

template <typename Scalar>
int BarrierEnergy<Scalar>::NNZ(const SolverData<Scalar>& solverData) const { return solverData.numVerts * 9; }

template <typename Scalar>
BarrierEnergy<Scalar>::BarrierEnergy(const SolverData<Scalar>& solverData, int& hessianIdxOffset, Scalar dhat) :dhat(dhat), Energy<Scalar>(hessianIdxOffset)
{
    hessianIdxOffset += NNZ(solverData);
}

template <typename Scalar>
Scalar BarrierEnergy<Scalar>::Val(const glm::tvec3<Scalar>* Xs, const SolverData<Scalar>& solverData) const {
    return 0;
}

template<typename Scalar>
void BarrierEnergy<Scalar>::Gradient(Scalar* grad, const SolverData<Scalar>& solverData, Scalar coef) const
{
}

template <typename Scalar>
void BarrierEnergy<Scalar>::Hessian(const SolverData<Scalar>& solverData, Scalar coef) const
{
}

template<typename Scalar>
Scalar BarrierEnergy<Scalar>::InitStepSize(const SolverData<Scalar>& solverData, Scalar* p) const
{
    return 1;
}

template class BarrierEnergy<float>;
template class BarrierEnergy<double>;