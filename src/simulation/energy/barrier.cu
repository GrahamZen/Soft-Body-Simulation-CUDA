#include <energy/barrier.h>
#include <collision/aabb.h>
#include <solverUtil.cuh>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

namespace Barrier {
    template <typename Scalar>
    __global__ void GradientKern(Scalar* grad, const glm::tvec3<Scalar>* Xs, const Query* queries, int numQueries) {
        int qIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (qIdx >= numQueries) return;
        const Query& q = queries[qIdx];
        glm::tvec3<Scalar> x0 = Xs[q.v0], x1 = Xs[q.v1], x2 = Xs[q.v2], x3 = Xs[q.v3];
        if (q.type == QueryType::EE) {
            q.dType = edge_edge_distance_type(x0, x1, x2, x3);
        }
        else if (q.type == QueryType::VF) {
            q.dType = point_triangle_distance_type(x0, x1, x2, x3);
        }
    }

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
    int num_queries = solverData.numQueries();
    if (num_queries == 0)return 0;
    Query* queries = solverData.queries();
    Scalar dhat = this->dhat;
    Scalar kappa = this->kappa;
    Scalar sum = thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(num_queries),
        [=] __host__ __device__(indexType qIdx) {
        Scalar d = queries[qIdx].d;
        if (d < dhat) {
            return barrierFunc(d, dhat, kappa, (Scalar)100.0);
        }
        else
            return (Scalar)0;
    },
        0.0,
        thrust::plus<Scalar>());
    return sum;
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