#include <energy/gravity.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

namespace Gravity {
    template <typename Scalar>
    __global__ void GradientKernel(Scalar* grad, const Scalar* mass, int numVerts, Scalar g, Scalar coef) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numVerts) return;
        grad[idx * 3 + 1] += coef * g * mass[idx];
    }
}

template<typename Scalar>
inline int GravityEnergy<Scalar>::NNZ(const SolverData<Scalar>& solverData) const
{
    return 0;
}

template<typename Scalar>
inline Scalar GravityEnergy<Scalar>::Val(const glm::tvec3<Scalar>* Xs, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams) const
{
    Scalar sum = thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(solverData.numVerts),
        [=] __host__ __device__(indexType vertIdx) {
        return Xs[vertIdx].y * solverData.mass[vertIdx];
    },
        0.0,
        thrust::plus<Scalar>());
    return g * sum;
}

template<typename Scalar>
inline void GravityEnergy<Scalar>::Gradient(Scalar* grad, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef) const
{
    int blockSize = 256;
    int numBlocks = (solverData.numVerts + blockSize - 1) / blockSize;
    Gravity::GradientKernel << <numBlocks, blockSize >> > (grad, solverData.mass, solverData.numVerts, g, coef);
}

template class GravityEnergy<float>;
template class GravityEnergy<double>;