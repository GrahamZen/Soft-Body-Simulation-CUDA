#include <energy/ip.h>
#include <energy/corotated.h>
#include <collision/bvh.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

struct AbsMax {
    __host__ __device__ double operator()(const double& a, const double& b) const {
        return max(abs(a), abs(b));
    }
};

struct AbsOp {
    __host__ __device__ double operator()(const double& x) const {
        return abs(x);
    }
};

IPEnergy::IPEnergy(const SolverData<double>& solverData) : inertia(solverData, nnz, solverData.numVerts, solverData.mass),
elastic(new CorotatedEnergy<double>(solverData, nnz)), implicitBarrier(solverData, nnz), barrier(solverData, nnz)
{
    hessianCapacity = nnz;
    cudaMalloc((void**)&gradient, sizeof(double) * solverData.numVerts * 3);
    cudaMalloc((void**)&hessianVal, sizeof(double) * hessianCapacity);
    cudaMalloc((void**)&hessianRowIdx, sizeof(int) * hessianCapacity);
    cudaMalloc((void**)&hessianColIdx, sizeof(int) * hessianCapacity);

    inertia.SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
    implicitBarrier.SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
    elastic->SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
    barrier.SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
}

IPEnergy::~IPEnergy()
{
    cudaFree(gradient);
    cudaFree(hessianVal);
    cudaFree(hessianRowIdx);
    cudaFree(hessianColIdx);
    if (elastic) delete elastic;
}

double IPEnergy::Val(const glm::dvec3* Xs, const SolverData<double>& solverData, const SolverParams<double>& solverParams, double h2) const
{
    return inertia.Val(Xs, solverData, solverParams) + h2 * (gravity.Val(Xs, solverData, solverParams) + elastic->Val(Xs, solverData, solverParams) + implicitBarrier.Val(Xs, solverData, solverParams) + barrier.Val(Xs, solverData, solverParams));
}

void IPEnergy::GradientHessian(const SolverData<double>& solverData, const SolverParams<double>& solverParams, double h2)
{
    int currentNNZ = NNZ(solverData);
    if (currentNNZ > hessianCapacity) {
        cudaFree(hessianVal);
        cudaFree(hessianRowIdx);
        cudaFree(hessianColIdx);
        hessianCapacity = static_cast<int>(currentNNZ * 1.5);

        cudaMalloc((void**)&hessianVal, sizeof(double) * hessianCapacity);
        cudaMalloc((void**)&hessianRowIdx, sizeof(int) * hessianCapacity);
        cudaMalloc((void**)&hessianColIdx, sizeof(int) * hessianCapacity);

        inertia.SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
        implicitBarrier.SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
        elastic->SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
        barrier.SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
    }

    cudaMemset(gradient, 0, sizeof(double) * solverData.numVerts * 3);
    cudaMemset(hessianVal, 0, sizeof(double) * currentNNZ);
    cudaMemset(hessianRowIdx, 0, sizeof(int) * currentNNZ);
    cudaMemset(hessianColIdx, 0, sizeof(int) * currentNNZ);
    inertia.GradientHessian(gradient, solverData, solverParams, 1);
    gravity.Gradient(gradient, solverData, solverParams, h2);
    elastic->GradientHessian(gradient, solverData, solverParams, h2);
    implicitBarrier.GradientHessian(gradient, solverData, solverParams, h2);
    barrier.GradientHessian(gradient, solverData, solverParams, h2);
}

void IPEnergy::UpdateKappa(SolverData<double>& solverData, const SolverParams<double>& solverParams, double h2) const
{
    solverData.kappa = 1.0;
    cudaMemset(gradient, 0, sizeof(double) * solverData.numVerts * 3);
    inertia.Gradient(gradient, solverData, solverParams, 1);
    gravity.Gradient(gradient, solverData, solverParams, h2);
    elastic->Gradient(gradient, solverData, solverParams, h2);
    implicitBarrier.Gradient(gradient, solverData, solverParams, h2);
    thrust::device_ptr<double> ptr_grad(gradient);
    double max_grad_elastic = thrust::transform_reduce(
        ptr_grad,
        ptr_grad + solverData.numVerts * 3,
        AbsOp(),
        0.0,
        AbsMax()
    );
    cudaMemset(gradient, 0, sizeof(double) * solverData.numVerts * 3);
    barrier.Gradient(gradient, solverData, solverParams, h2);
    double max_grad_barrier = thrust::transform_reduce(
        ptr_grad,
        ptr_grad + solverData.numVerts * 3,
        AbsOp(),
        0.0,
        AbsMax()
    );
    double min_kappa = 100;
    if (max_grad_barrier > 1e-9) {
        double computed_kappa = max_grad_elastic / max_grad_barrier;
        solverData.kappa = std::max(min_kappa, computed_kappa);
    }
    else {
        solverData.kappa = min_kappa;
    }
}

double IPEnergy::InitStepSize(SolverData<double>& solverData, const SolverParams<double>& solverParams, double* p, glm::tvec3<double>* XTmp) const
{
    double step = 0.95 * std::min(implicitBarrier.InitStepSize(solverData, p), barrier.InitStepSize(solverData, p, XTmp));
    if (step < 1e-12) {
        return 0.0;
    }
    return std::min(1.0, step);
}

int IPEnergy::NNZ(const SolverData<double>& solverData) const
{
    return inertia.NNZ(solverData) + implicitBarrier.NNZ(solverData) + elastic->NNZ(solverData) + barrier.NNZ(solverData);
}