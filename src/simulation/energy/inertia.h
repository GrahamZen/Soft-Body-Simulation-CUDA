#pragma once

#include <def.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <cuda/functional>

template <typename HighP>
class InertiaEnergy {
public:
    InertiaEnergy(int numVerts, const HighP* dev_mass);
    HighP Val(const glm::tvec3<HighP>* dev_x, const glm::tvec3<HighP>* dev_xTilde, const HighP* mass, int numVerts) const;
    void Gradient(HighP* grad, const glm::tvec3<HighP>* dev_x, const glm::tvec3<HighP>* dev_xTilde, const HighP* dev_mass, int numVerts) const;
    void Hessian(HighP*& hessianVal, int*& hessianRowIdx, int*& hessianColIdx) const;
private:
    int numVerts = 0;
    HighP* hessianVal = nullptr;
    int* hessianRowIdx = nullptr;
    int* hessianColIdx = nullptr;
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
        int offset = idx * 3;
        gradient[offset] = dev_mass[idx] * (dev_x[offset] - dev_xTilde[offset]);
        gradient[offset + 1] = dev_mass[idx] * (dev_x[offset + 1] - dev_xTilde[offset + 1]);
        gradient[offset + 2] = dev_mass[idx] * (dev_x[offset + 2] - dev_xTilde[offset + 2]);
    }
}

template <typename HighP>
InertiaEnergy<HighP>::InertiaEnergy(int numVerts, const HighP* dev_mass) : numVerts(numVerts) {
    cudaMalloc(&hessianVal, sizeof(HighP) * numVerts * 3);
    cudaMalloc(&hessianRowIdx, sizeof(HighP) * numVerts * 3);
    cudaMalloc(&hessianColIdx, sizeof(HighP) * numVerts * 3);

    int threadsPerBlock = 256;
    int numBlocks = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    Inertia::hessianKern << <numBlocks, threadsPerBlock >> > (dev_mass, hessianVal, hessianRowIdx, hessianColIdx, numVerts);
}

template <typename HighP>
HighP InertiaEnergy<HighP>::Val(const glm::tvec3<HighP>* dev_x, const glm::tvec3<HighP>* dev_xTilde, const HighP* dev_mass, int numVerts) const {
    // ||m(x - x_tilde)||^2 * 0.5.
    thrust::device_ptr<const HighP> dev_ptrX(dev_x);
    thrust::device_ptr<const HighP> dev_ptrXTilde(dev_xTilde);
    thrust::device_ptr<const HighP> dev_ptrMass(dev_mass);
    HighP sum = thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple(dev_ptrX, dev_ptrXTilde, dev_ptrMass)),
        thrust::make_zip_iterator(thrust::make_tuple(dev_ptrX + numVerts * 3, dev_ptrXTilde + numVerts * 3, dev_ptrMass + numVerts)),
        []__host__ __device__(const thrust::tuple<HighP, HighP, HighP>&t) {
        HighP x = thrust::get<0>(t);
        HighP xTilde = thrust::get<1>(t);
        HighP mass = thrust::get<2>(t);
        HighP diff = x - xTilde;
        return mass * diff * diff;
    },
        0.0,
        thrust::plus<HighP>());
    return sum * 0.5;
}

template <typename HighP>
void InertiaEnergy<HighP>::Gradient(HighP* dev_grad, const glm::tvec3<HighP>* dev_x, const glm::tvec3<HighP>* dev_xTilde, const HighP* dev_mass, int numVerts) const {
    // m(x - x_tilde).
    int threadsPerBlock = 256;
    int numBlocks = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    Inertia::gradientKern << <numBlocks, threadsPerBlock >> > (dev_x, dev_xTilde, dev_mass, dev_grad, numVerts);
}

template <typename HighP>
void InertiaEnergy<HighP>::Hessian(HighP*& hessianVal, int*& hessianRowIdx, int*& hessianColIdx) const
{
    hessianVal = this->hessianVal;
    hessianRowIdx = this->hessianRowIdx;
    hessianColIdx = this->hessianColIdx;
}
