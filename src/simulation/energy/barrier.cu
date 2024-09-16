#include <energy/barrier.h>
#include <collision/bvh.h>
#include <solverUtil.cuh>
#include <matrix.h>
#include <distance/distance_type.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

namespace Barrier {
    template <typename Scalar>
    __global__ void GradientKern(Scalar* grad, const glm::tvec3<Scalar>* Xs, const Query* queries, int numQueries, Scalar dhat, Scalar coef) {
        int qIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (qIdx >= numQueries) return;
        const Query& q = queries[qIdx];
        if (q.d > dhat) return;
        glm::tvec3<Scalar> x0 = Xs[q.v0], x1 = Xs[q.v1], x2 = Xs[q.v2], x3 = Xs[q.v3];
        Vector12<Scalar> localGrad;
        if (q.type == QueryType::EE) {
            localGrad = coef * ipc::edge_edge_distance_gradient(x0, x1, x2, x3, q.dType);
        }
        else if (q.type == QueryType::VF) {
            localGrad = coef * ipc::point_triangle_distance_gradient(x0, x1, x2, x3, q.dType);
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
    template <typename Scalar>
    __global__ void hessianKern(Scalar* hessianVal, int* hessianRowIdx, int* hessianColIdx, const glm::tvec3<Scalar>* Xs, const Query* queries, int numQueries, Scalar dhat, Scalar coef) {
        int qIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (qIdx >= numQueries) return;
        const Query& q = queries[qIdx];
        if (q.d > dhat) return;
        glm::tvec3<Scalar> x0 = Xs[q.v0], x1 = Xs[q.v1], x2 = Xs[q.v2], x3 = Xs[q.v3];
        Matrix12<Scalar> localHess;
        if (q.type == QueryType::EE) {
            localHess = coef * ipc::edge_edge_distance_hessian(x0, x1, x2, x3, q.dType);
        }
        else if (q.type == QueryType::VF) {
            localHess = coef * ipc::point_triangle_distance_hessian(x0, x1, x2, x3, q.dType);
        }
        indexType v[4] = { q.v0, q.v1, q.v2, q.v3 };
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int row = v[i] * 3, col = v[j] * 3;
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        int idx = row + k;
                        int jdx = col + l;
                        int hIdx = qIdx * 144 + i * 36 + j * 9 + k * 3 + l;
                        hessianVal[hIdx] = localHess[i * 3 + k][j * 3 + l];
                        hessianRowIdx[hIdx] = idx;
                        hessianColIdx[hIdx] = jdx;
                    }
                }
            }
        }

    }
}

template <typename Scalar>
int BarrierEnergy<Scalar>::NNZ(const SolverData<Scalar>& solverData) {
    Energy<Scalar>::nnz = solverData.numQueries() * 9;
    return Energy<Scalar>::nnz;
}

template <typename Scalar>
BarrierEnergy<Scalar>::BarrierEnergy(const SolverData<Scalar>& solverData, int& hessianIdxOffset, Scalar dhat) :dhat(dhat), Energy<Scalar>(hessianIdxOffset)
{
    hessianIdxOffset += solverData.pCollisionDetection->GetMaxNumQueries() * 144;
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
    Barrier::GradientKern << <numBlocks, threadsPerBlock >> > (grad, solverData.X, solverData.queries(), numQueries, dhat, coef);
}

template <typename Scalar>
void BarrierEnergy<Scalar>::Hessian(const SolverData<Scalar>& solverData, Scalar coef) const
{
    int numQueries = solverData.numQueries();
    if (numQueries == 0)return;
    int threadsPerBlock = 256;
    int numBlocks = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    Barrier::hessianKern << <numBlocks, threadsPerBlock >> > (hessianVal, hessianRowIdx, hessianColIdx, solverData.X, solverData.queries(), numQueries, dhat, coef);
}

template<typename Scalar>
Scalar BarrierEnergy<Scalar>::InitStepSize(const SolverData<Scalar>& solverData, Scalar* p, glm::tvec3<Scalar>* XTmp) const
{
    return solverData.pCollisionDetection->ComputeMinStepSize(solverData.numVerts, solverData.numTris, solverData.Tri, solverData.X, XTmp, solverData.dev_TriFathers, true);
}

template class BarrierEnergy<float>;
template class BarrierEnergy<double>;