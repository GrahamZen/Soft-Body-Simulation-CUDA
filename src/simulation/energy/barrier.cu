#include <energy/barrier.h>
#include <collision/bvh.h>
#include <solverUtil.cuh>
#include <matrix.h>
#include <distance/distance_type.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

namespace Barrier {
    template <typename Scalar>
    __forceinline__ __host__ __device__ Scalar barrierSquareFunc(Scalar d_sqr, Scalar dhat, Scalar kappa) {
        Scalar s = d_sqr / (dhat * dhat);
        return 0.5 * dhat * kappa * 0.125 * (s - 1) * log(s);
    }

    template <typename Scalar>
    __forceinline__ __host__ __device__ Scalar barrierSquareFuncDerivative(Scalar d_sqr, Scalar dhat, Scalar kappa) {
        Scalar dhat_sqr = dhat * dhat;
        Scalar s = d_sqr / dhat_sqr;
        return 0.5 * dhat * (kappa / 8 * (log(s) / dhat_sqr + (s - 1) / d_sqr));
    }

    template <typename Scalar>
    __forceinline__ __host__ __device__ Matrix12<Scalar> barrierSquareFuncHess(Scalar d_sqr, Scalar dhat, Scalar kappa, const Vector12<Scalar>& d_sqr_grad, const Matrix12<Scalar>& d_sqr_hess) {
        Scalar dhat_sqr = dhat * dhat;
        Scalar s = d_sqr / dhat_sqr;
        return 0.5 * dhat * (kappa / (8 * d_sqr * d_sqr * dhat_sqr) * (d_sqr + dhat_sqr) * Matrix12<Scalar>(d_sqr_grad, d_sqr_grad)
            + (kappa / 8 * (log(s) / dhat_sqr + (s - 1) / d_sqr)) * d_sqr_hess);
    }

    template <typename Scalar>
    __global__ void GradientKern(Scalar* grad, const glm::tvec3<Scalar>* Xs, const Query* queries, int numQueries, Scalar dhat, Scalar kappa, Scalar coef) {
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
        localGrad = coef * barrierSquareFuncDerivative((Scalar)q.d, dhat, kappa) * localGrad;
        atomicAdd(&grad[q.v0 * 3 + 0], localGrad[0]);
        atomicAdd(&grad[q.v0 * 3 + 1], localGrad[1]);
        atomicAdd(&grad[q.v0 * 3 + 2], localGrad[2]);
        atomicAdd(&grad[q.v1 * 3 + 0], localGrad[3]);
        atomicAdd(&grad[q.v1 * 3 + 1], localGrad[4]);
        atomicAdd(&grad[q.v1 * 3 + 2], localGrad[5]);
        atomicAdd(&grad[q.v2 * 3 + 0], localGrad[6]);
        atomicAdd(&grad[q.v2 * 3 + 1], localGrad[7]);
        atomicAdd(&grad[q.v2 * 3 + 2], localGrad[8]);
        atomicAdd(&grad[q.v3 * 3 + 0], localGrad[9]);
        atomicAdd(&grad[q.v3 * 3 + 1], localGrad[10]);
        atomicAdd(&grad[q.v3 * 3 + 2], localGrad[11]);
    }
    template <typename Scalar>
    __global__ void hessianKern(Scalar* hessianVal, int* hessianRowIdx, int* hessianColIdx, const glm::tvec3<Scalar>* Xs, const Query* queries, int numQueries, Scalar dhat, Scalar kappa, Scalar coef) {
        int qIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (qIdx >= numQueries) return;
        const Query& q = queries[qIdx];
        if (q.d > dhat) return;
        glm::tvec3<Scalar> x0 = Xs[q.v0], x1 = Xs[q.v1], x2 = Xs[q.v2], x3 = Xs[q.v3];
        Vector12<Scalar> localGrad;
        Matrix12<Scalar> localHess;
        if (q.type == QueryType::EE) {
            localGrad = ipc::edge_edge_distance_gradient(x0, x1, x2, x3, q.dType);
            localHess = ipc::edge_edge_distance_hessian(x0, x1, x2, x3, q.dType);
        }
        else if (q.type == QueryType::VF) {
            localGrad = ipc::point_triangle_distance_gradient(x0, x1, x2, x3, q.dType);
            localHess = ipc::point_triangle_distance_hessian(x0, x1, x2, x3, q.dType);
        }
        localHess = coef * barrierSquareFuncHess((Scalar)q.d, dhat, kappa, localGrad, localHess);
        indexType v[4] = { q.v0, q.v1, q.v2, q.v3 };
        for (int i = 0; i < 4; i++) {
            int row = v[i] * 3;
            for (int j = 0; j < 4; j++) {
                int col = v[j] * 3;
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        int idx = qIdx * 144 + (i * 4 + j) * 9 + k * 3 + l;
                        hessianVal[idx] = localHess[i * 3 + k][j * 3 + l];
                        hessianRowIdx[idx] = row + k;
                        hessianColIdx[idx] = col + l;
                    }
                }
            }
        }
    }
}

template <typename Scalar>
int BarrierEnergy<Scalar>::NNZ(const SolverData<Scalar>& solverData) const {
    return solverData.numQueries() * 144;
}

template <typename Scalar>
BarrierEnergy<Scalar>::BarrierEnergy(const SolverData<Scalar>& solverData, int& hessianIdxOffset, Scalar dhat, Scalar kappa)
    :dhat(dhat), kappa(kappa), Energy<Scalar>(hessianIdxOffset)
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
            return Barrier::barrierSquareFunc(d, dhat, kappa);
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
    Barrier::GradientKern << <numBlocks, threadsPerBlock >> > (grad, solverData.X, solverData.queries(), numQueries, dhat, (Scalar)kappa, coef);
}

template <typename Scalar>
void BarrierEnergy<Scalar>::Hessian(const SolverData<Scalar>& solverData, Scalar coef) const
{
    int numQueries = solverData.numQueries();
    if (numQueries == 0)return;
    int threadsPerBlock = 256;
    int numBlocks = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    Barrier::hessianKern << <numBlocks, threadsPerBlock >> > (hessianVal, hessianRowIdx, hessianColIdx, solverData.X, solverData.queries(), numQueries, dhat, (Scalar)kappa, coef);
}

template<typename Scalar>
Scalar BarrierEnergy<Scalar>::InitStepSize(const SolverData<Scalar>& solverData, Scalar* p, glm::tvec3<Scalar>* XTmp) const
{
    return solverData.pCollisionDetection->ComputeMinStepSize(solverData.numVerts, solverData.numTris, solverData.Tri, solverData.X, XTmp, solverData.dev_TriFathers, true);
}

template class BarrierEnergy<float>;
template class BarrierEnergy<double>;