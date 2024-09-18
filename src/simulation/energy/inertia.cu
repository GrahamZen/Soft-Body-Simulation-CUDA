#include <energy/inertia.h>
#include <glm/gtx/norm.hpp>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

namespace Inertia {
    template <typename Scalar>
    __global__ void hessianKern(const Scalar* dev_mass, Scalar* hessianVal, int* hessianRowIdx, int* hessianColIdx, int numVerts) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numVerts) {
            return;
        }
        int offset = idx * 3;
        hessianVal[offset] = dev_mass[idx];
        hessianRowIdx[offset] = offset;
        hessianColIdx[offset] = offset;
        hessianVal[offset + 1] = dev_mass[idx];
        hessianRowIdx[offset + 1] = offset + 1;
        hessianColIdx[offset + 1] = offset + 1;
        hessianVal[offset + 2] = dev_mass[idx];
        hessianRowIdx[offset + 2] = offset + 2;
        hessianColIdx[offset + 2] = offset + 2;
    }

    template <typename Scalar>
    __global__ void gradientKern(const glm::tvec3<Scalar>* X, const glm::tvec3<Scalar>* XTilde, const Scalar* dev_mass, Scalar* gradient, int numVerts) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numVerts) {
            return;
        }
        glm::tvec3<Scalar> diff = dev_mass[idx] * (X[idx] - XTilde[idx]);
        gradient[idx * 3] += diff.x;
        gradient[idx * 3 + 1] += diff.y;
        gradient[idx * 3 + 2] += diff.z;
    }
}

template <typename Scalar>
int InertiaEnergy<Scalar>::NNZ(const SolverData<Scalar>& solverData) const {
    return solverData.numVerts * 3;
}

template <typename Scalar>
InertiaEnergy<Scalar>::InertiaEnergy(const SolverData<Scalar>& solverData, int& hessianIdxOffset, int numVerts, const Scalar* dev_mass) :
    Energy<Scalar>(hessianIdxOffset)
{
    hessianIdxOffset += NNZ(solverData);
}

template <typename Scalar>
Scalar InertiaEnergy<Scalar>::Val(const glm::tvec3<Scalar>* Xs, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams) const {
    // ||(x - x_tilde)||m^2 * 0.5.
    Scalar sum = thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(solverData.numVerts),
        [=]__host__ __device__(indexType vertIdx) {
        glm::tvec3<Scalar> diff = Xs[vertIdx] - solverData.XTilde[vertIdx];
        Scalar mass = solverData.mass[vertIdx];
        return mass * glm::dot(diff, diff);
    },
        0.0,
        thrust::plus<Scalar>());
    return sum * 0.5;
}

template<typename Scalar>
void InertiaEnergy<Scalar>::Gradient(Scalar* grad, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef) const
{
    // m(x - x_tilde).
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    Inertia::gradientKern << <numBlocks, threadsPerBlock >> > (solverData.X, solverData.XTilde, solverData.mass, grad, solverData.numVerts);
}

template <typename Scalar>
void InertiaEnergy<Scalar>::Hessian(const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef) const
{
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    Inertia::hessianKern << <numBlocks, threadsPerBlock >> > (solverData.mass, hessianVal, hessianRowIdx, hessianColIdx, solverData.numVerts);
}

template class InertiaEnergy<float>;
template class InertiaEnergy<double>;