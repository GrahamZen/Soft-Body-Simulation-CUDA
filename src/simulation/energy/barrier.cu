#include <energy/barrier.h>
#include <collision/aabb.h>
#include <solverUtil.cuh>
#include <vector.h>
#include <distance/point_triangle.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

namespace Barrier {
    template <typename Scalar>
    __global__ void GradientKern(Scalar* grad, const glm::tvec3<Scalar>* Xs, const Query* queries, int numQueries, Scalar dhat) {
        int qIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (qIdx >= numQueries) return;
        const Query& q = queries[qIdx];
        if (q.d > dhat) return;
        glm::tvec3<Scalar> x0 = Xs[q.v0], x1 = Xs[q.v1], x2 = Xs[q.v2], x3 = Xs[q.v3];
        Vector12<Scalar> localGrad;
        if (q.type == QueryType::EE) {
            localGrad = ipc::edge_edge_distance_gradient(x0, x1, x2, x3, q.dType);
        }
        else if (q.type == QueryType::VF) {
            localGrad = ipc::point_triangle_distance_gradient(x0, x1, x2, x3, q.dType);
        }
        grad[q.v0 * 3 + 0] += localGrad[0];
        grad[q.v0 * 3 + 1] += localGrad[1];
        grad[q.v0 * 3 + 2] += localGrad[2];
        grad[q.v1 * 3 + 0] += localGrad[3];
        grad[q.v1 * 3 + 1] += localGrad[4];
        grad[q.v1 * 3 + 2] += localGrad[5];
        grad[q.v2 * 3 + 0] += localGrad[6];
        grad[q.v2 * 3 + 1] += localGrad[7];
        grad[q.v2 * 3 + 2] += localGrad[8];
        grad[q.v3 * 3 + 0] += localGrad[9];
        grad[q.v3 * 3 + 1] += localGrad[10];
        grad[q.v3 * 3 + 2] += localGrad[11];
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
    int numQueries = solverData.numQueries();
    if (numQueries == 0)return;
    int threadsPerBlock = 256;
    int numBlocks = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    Barrier::GradientKern << <numBlocks, threadsPerBlock >> > (grad, solverData.X, solverData.queries(), numQueries, dhat);
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