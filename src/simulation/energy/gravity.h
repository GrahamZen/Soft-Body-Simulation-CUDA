#pragma once

#include <def.h>
#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

template <typename HighP>
class GravityEnergy : public Energy<HighP> {
public:
    GravityEnergy() = default;
    virtual int NNZ(const SolverData<HighP>& solverData) const override;
    virtual HighP Val(const glm::tvec3<HighP>* Xs, const SolverData<HighP>& solverData) const override;
    virtual void Gradient(HighP* grad, const SolverData<HighP>& solverData, HighP coef) const override;
    virtual void Hessian(const SolverData<HighP>& solverData, HighP coef) const override {}
    const HighP g = 9.8;
};

namespace Gravity {
    template <typename HighP>
    __global__ void GradientKernel(HighP* grad, const HighP* mass, int numVerts, HighP g, HighP coef) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numVerts) return;
        grad[idx * 3 + 1] += coef * g * mass[idx];
    }
}

template<typename HighP>
inline int GravityEnergy<HighP>::NNZ(const SolverData<HighP>& solverData) const
{
    return 0;
}

template<typename HighP>
inline HighP GravityEnergy<HighP>::Val(const glm::tvec3<HighP>* Xs, const SolverData<HighP>& solverData) const
{
    HighP sum = thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(solverData.numVerts),
        [=] __host__ __device__(indexType vertIdx) {
        return Xs[vertIdx].y * solverData.mass[vertIdx];
    },
        0.0,
        thrust::plus<HighP>());
    return g * sum;
}

template<typename HighP>
inline void GravityEnergy<HighP>::Gradient(HighP* grad, const SolverData<HighP>& solverData, HighP coef) const
{
    int blockSize = 256;
    int numBlocks = (solverData.numVerts + blockSize - 1) / blockSize;
    Gravity::GradientKernel << <numBlocks, blockSize >> > (grad, solverData.mass, solverData.numVerts, g, coef);
}
