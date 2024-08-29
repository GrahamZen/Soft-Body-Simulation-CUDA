#pragma once

#include <energy/energy.h>
#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>
#include <cuda/functional>
#include <glm/gtx/norm.hpp>

template <typename HighP>
class InertiaEnergy : public Energy<HighP> {
public:
    InertiaEnergy(const SolverData<HighP>& solverData, int& hessianIdxOffset, int numVerts, const HighP* dev_mass);
    virtual ~InertiaEnergy() = default;
    virtual int NNZ(const SolverData<HighP>& solverData) const override;
    virtual HighP Val(const glm::tvec3<HighP>* Xs, const SolverData<HighP>& solverData, HighP coef) const override;
    virtual void Gradient(HighP* grad, const SolverData<HighP>& solverData, HighP coef) const override;
    virtual void Hessian(const SolverData<HighP>& solverData, HighP coef) const override;
private:
    int numVerts = 0;
};
namespace Inertia {
    template <typename HighP>
    __global__ void hessianKern(const HighP* dev_mass, HighP* hessianVal, int* hessianRowIdx, int* hessianColIdx, int numVerts) {
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

    template <typename HighP>
    __global__ void gradientKern(const glm::tvec3<HighP>* dev_x, const glm::tvec3<HighP>* dev_xTilde, const HighP* dev_mass, HighP* gradient, int numVerts) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numVerts) {
            return;
        }
        glm::tvec3<HighP> x = dev_x[idx];
        glm::tvec3<HighP> xTilde = dev_xTilde[idx];
        HighP mass = dev_mass[idx];
        gradient[idx * 3] += mass * (x.x - xTilde.x);
        gradient[idx * 3 + 1] += mass * (x.y - xTilde.y);
        gradient[idx * 3 + 2] += mass * (x.z - xTilde.z);
    }
}

template <typename HighP>
int InertiaEnergy<HighP>::NNZ(const SolverData<HighP>& solverData) const { return solverData.numVerts * 3; }

template <typename HighP>
InertiaEnergy<HighP>::InertiaEnergy(const SolverData<HighP>& solverData, int& hessianIdxOffset, int numVerts, const HighP* dev_mass) :
    Energy<HighP>(hessianIdxOffset), numVerts(numVerts)
{
    hessianIdxOffset += NNZ(solverData);
}

template <typename HighP>
HighP InertiaEnergy<HighP>::Val(const glm::tvec3<HighP>* Xs, const SolverData<HighP>& solverData, HighP coef) const {
    // ||(x - x_tilde)||m^2 * 0.5.
    HighP sum = thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(solverData.numVerts),
        [=]__host__ __device__(indexType vertIdx) {
        glm::tvec3<HighP> diff = Xs[vertIdx] - solverData.XTilde[vertIdx];
        HighP mass = solverData.mass[vertIdx];
        return mass * glm::dot(diff, diff);
    },
        0.0,
        thrust::plus<HighP>());
    return sum * 0.5;
}

template<typename HighP>
void InertiaEnergy<HighP>::Gradient(HighP* grad, const SolverData<HighP>& solverData, HighP coef) const
{
    // m(x - x_tilde).
    int threadsPerBlock = 256;
    int numBlocks = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    Inertia::gradientKern << <numBlocks, threadsPerBlock >> > (solverData.X, solverData.XTilde, solverData.mass, grad, solverData.numVerts);
}

template <typename HighP>
void InertiaEnergy<HighP>::Hessian(const SolverData<HighP>& solverData, HighP coef) const
{
    int threadsPerBlock = 256;
    int numBlocks = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    Inertia::hessianKern << <numBlocks, threadsPerBlock >> > (solverData.mass, hessianVal, hessianRowIdx, hessianColIdx, numVerts);
}
