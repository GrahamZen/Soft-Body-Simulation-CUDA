#include <energy/elasticity.h>
#include <solver/solverUtil.cuh>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

template <typename HighP>
class CorotatedEnergy : public ElasticEnergy<HighP> {
public:
    CorotatedEnergy() = default;
    virtual ~CorotatedEnergy() override = default;
    virtual HighP Val(const SolverData<HighP>& solverData) const override;
    virtual void Gradient(HighP* grad, const SolverData<HighP>& solverData) const override;
    virtual void Hessian(HighP*& hessianVal, int*& hessianRowIdx, int*& hessianColIdx, const SolverData<HighP>& solverData) const override;
};

namespace Corotated {
    template <typename HighP>
    __global__ void GradientKern(HighP* grad, const glm::tvec3<HighP>* X, const indexType* Tet, const glm::tmat3x3<HighP>* inv_Dm, const indexType numTets) {
        indexType tetIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (tetIndex >= numTets) return;

        glm::tmat3x3<HighP> Ds = Build_Edge_Matrix(X, Tet, tetIndex);
        glm::tmat3x3<HighP> F = Ds * inv_Dm[tetIndex];
        glm::tmat3x3<HighP> U, S, V;

        svdGLM(F, U, S, V);
        glm::tmat3x3<HighP> R = U * glm::transpose(V);
        glm::tmat3x3<HighP> P = 2 * (F - R);
        glm::tmat3x3<HighP> dPsidx = P * inv_Dm[tetIndex];
        atomicAdd(&grad[Tet[tetIndex * 4 + 0] * 3 + 0], dPsidx[0][0]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 0] * 3 + 1], dPsidx[0][1]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 0] * 3 + 2], dPsidx[0][2]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 1] * 3 + 0], dPsidx[1][0]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 1] * 3 + 1], dPsidx[1][1]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 1] * 3 + 2], dPsidx[1][2]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 2] * 3 + 0], dPsidx[2][0]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 2] * 3 + 1], dPsidx[2][1]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 2] * 3 + 2], dPsidx[2][2]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 3] * 3 + 0], -dPsidx[0][0] - dPsidx[1][0] - dPsidx[2][0]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 3] * 3 + 1], -dPsidx[0][1] - dPsidx[1][1] - dPsidx[2][1]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 3] * 3 + 2], -dPsidx[0][2] - dPsidx[1][2] - dPsidx[2][2]);
    }
    template <typename HighP>
    __global__ void HessianKern(HighP* hessianVal, int* hessianRowIdx, int* hessianColIdx, const glm::tvec3<HighP>* X, const indexType* Tet, const glm::tmat3x3<HighP>* inv_Dm, const indexType numTets) {
        indexType tetIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (tetIndex >= numTets) return;

        glm::tmat3x3<HighP> Ds = Build_Edge_Matrix(X, Tet, tetIndex);
        glm::tmat3x3<HighP> F = Ds * inv_Dm[tetIndex];
        glm::tmat3x3<HighP> U, S, V;

        svdGLM(F, U, S, V);
        glm::tmat3x3<HighP> R = U * glm::transpose(V);
        glm::tmat3x3<HighP> P = 2 * (F - R);
    }
}


template <typename HighP>
HighP CorotatedEnergy<HighP>::Val(const SolverData<HighP>& solverData) const {
    thrust::device_ptr<glm::tvec3<HighP>> dev_ptr(solverData.dev_x);
    thrust::device_ptr<indexType> dev_tet(solverData.dev_tet);
    thrust::device_ptr<glm::tmat3x3<HighP>> dev_inv_Dm(solverData.dev_inv_Dm);

    HighP sum = thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(solverData.numTets),
        [=] __device__(indexType tetIndex) {
        glm::tmat3x3<HighP> Ds = Build_Edge_Matrix(solverData.X, solverData.Tet, tetIndex);
        glm::tmat3x3<HighP> V = Ds * dev_inv_Dm[tetIndex];
        glm::tmat3x3<HighP> U;
        glm::tmat3x3<HighP> S;

        svdGLM(V, U, S, V);
        return frobeniusNorm(S - glm::tmat3x3<HighP>(1)) * 0.5;
    },
        (HighP)0,
        thrust::plus<HighP>()
    );
    return sum;
}

template <typename HighP>
void CorotatedEnergy<HighP>::Gradient(HighP* grad, const SolverData<HighP>& solverData) const {
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    Corotated::GradientKern << <numBlocks, threadsPerBlock >> > (grad, solverData.dev_x, solverData.dev_tet, solverData.dev_inv_Dm, solverData.numTets);
}

template <typename HighP>
void CorotatedEnergy<HighP>::Hessian(HighP*& hessianVal, int*& hessianRowIdx, int*& hessianColIdx, const SolverData<HighP>& solverData) const {
}