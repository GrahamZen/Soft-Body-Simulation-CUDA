#include <cuda_runtime.h>
#include <energy/inertia.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <cuda/functional>
__global__ void hessianKern(const double* dev_mass, double* hessianVal, int* hessianRowIdx, int* hessianColIdx, int numVerts) {
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

__global__ void gradientKern(const double* dev_x, const double* dev_xTilde, const double* dev_mass, double* gradient, int numVerts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVerts) {
        return;
    }
    int offset = idx * 3;
    gradient[offset] = dev_mass[idx] * (dev_x[offset] - dev_xTilde[offset]);
    gradient[offset + 1] = dev_mass[idx] * (dev_x[offset + 1] - dev_xTilde[offset + 1]);
    gradient[offset + 2] = dev_mass[idx] * (dev_x[offset + 2] - dev_xTilde[offset + 2]);
}


InertiaEnergy::InertiaEnergy(int numVerts, const double* dev_mass) : numVerts(numVerts) {
    cudaMalloc(&hessianVal, sizeof(double) * numVerts * 3);
    cudaMalloc(&hessianRowIdx, sizeof(double) * numVerts * 3);
    cudaMalloc(&hessianColIdx, sizeof(double) * numVerts * 3);

    int threadsPerBlock = 256;
    int numBlocks = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    hessianKern << <numBlocks, threadsPerBlock >> > (dev_mass, hessianVal, hessianRowIdx, hessianColIdx, numVerts);
}

double InertiaEnergy::Val(const double* dev_x, const double* dev_xTilde, const double* dev_mass, int numVerts) const {
    // ||m(x - x_tilde)||^2 * 0.5.
    thrust::device_ptr<const double> dev_ptrX(dev_x);
    thrust::device_ptr<const double> dev_ptrXTilde(dev_xTilde);
    thrust::device_ptr<const double> dev_ptrMass(dev_mass);
    double sum = thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple(dev_ptrX, dev_ptrXTilde, dev_ptrMass)),
        thrust::make_zip_iterator(thrust::make_tuple(dev_ptrX + numVerts * 3, dev_ptrXTilde + numVerts * 3, dev_ptrMass + numVerts)),
        []__host__ __device__(const thrust::tuple<double, double, double>&t) {
        double x = thrust::get<0>(t);
        double xTilde = thrust::get<1>(t);
        double mass = thrust::get<2>(t);
        double diff = x - xTilde;
        return mass * diff * diff;
    },
        0.0,
        thrust::plus<double>());
    return sum * 0.5;
}

void InertiaEnergy::Gradient(double* dev_grad, const double* dev_x, const double* dev_xTilde, const double* dev_mass, int numVerts) const {
    // m(x - x_tilde).
    int threadsPerBlock = 256;
    int numBlocks = (numVerts + threadsPerBlock - 1) / threadsPerBlock;
    gradientKern << <numBlocks, threadsPerBlock >> > (dev_x, dev_xTilde, dev_mass, dev_grad, numVerts);
}

void InertiaEnergy::Hessian(double*& hessianVal, int*& hessianRowIdx, int*& hessianColIdx) const
{
    hessianVal = this->hessianVal;
    hessianRowIdx = this->hessianRowIdx;
    hessianColIdx = this->hessianColIdx;
}
