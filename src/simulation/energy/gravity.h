#pragma once

#include <def.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>

namespace Gravity {
    template <typename HighP>
    __global__ void GradientKernel(HighP* grad, const glm::tvec3<HighP>* dev_x, const HighP* mass, int numVerts, HighP g) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numVerts) return;
        grad[idx * 3 + 1] = -g * mass[idx];
    }
}

template <typename HighP>
class GravityEnergy {
public:
    HighP Val(const glm::tvec3<HighP>* dev_x, const HighP* mass, int numVerts) const {
        thrust::device_ptr<HighP> dev_ptr(dev_x);
        thrust::device_ptr<HighP> dev_mass(mass);
        HighP sum = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(dev_ptr, dev_mass)),
            thrust::make_zip_iterator(thrust::make_tuple(dev_ptr + numVerts, dev_mass + numVerts)),
            [] __device__(const thrust::tuple<HighP, HighP>&t) {
            return thrust::get<0>(t).y * thrust::get<1>(t);
        },
            (HighP)0,
            thrust::plus<HighP>()
        );
        return -g * sum;
    }
    void Gradient(HighP* grad, const glm::tvec3<HighP>* dev_x, const HighP* mass, int numVerts) {
        int blockSize = 256;
        int numBlocks = (numVerts + blockSize - 1) / blockSize;
        Gravity::GradientKernel << <numBlocks, blockSize >> > (grad, dev_x, mass, numVerts, g);
    }
    const HighP g = 9.8;
};