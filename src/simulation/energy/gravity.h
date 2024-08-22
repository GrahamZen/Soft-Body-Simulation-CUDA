#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
template <typename T>
class GravityEnergy {
public:
    __global__ void GradientKernel(T* grad, const T* dev_x, const T* mass, int numVerts, T g){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numVerts) return;
        grad[idx] = -g * mass[idx];
    }
    T Val(const T* dev_x, const T* mass, int numVerts) const {
        thrust::device_ptr<T> dev_ptr(dev_x);
        thrust::device_ptr<T> dev_mass(mass);
        T sum = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(dev_ptr, dev_mass)),
            thrust::make_zip_iterator(thrust::make_tuple(dev_ptr + numVerts, dev_mass + numVerts)),
            [] __device__ (const thrust::tuple<T, T>& t){
                return -thrust::get<0>(t) * thrust::get<1>(t);
            },
            (T)0,
            thrust::plus<T>()
        );
        return -g * sum;
    }
    void Gradient(T* grad, const T* dev_x, const T* mass, int numVerts){
        int blockSize = 256;
        int numBlocks = (numVerts + blockSize - 1) / blockSize;
        GradientKernel<<<numBlocks, blockSize>>>(grad, dev_x, mass, numVerts, g);
    }
    const T g = 9.8;
};