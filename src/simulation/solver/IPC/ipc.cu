#include <IPC/ipc.h>
#include <linear/choleskyImmed.h>
#include <fixedBodyData.h>
#include <utilities.cuh>
#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>

IPCSolver::IPCSolver(int threadsPerBlock, const SolverData<double>& solverData, double tol)
    : numVerts(solverData.numVerts), tolerance(tol), FEMSolver(threadsPerBlock, solverData),
    energy(solverData), linearSolver(new CholeskySpImmedSolver<double>(solverData.numVerts * 3))
{
    cudaMalloc(&p, sizeof(double) * solverData.numVerts * 3);
    cudaMalloc(&xTmp, sizeof(glm::dvec3) * solverData.numVerts);
    cudaMalloc(&x_n, sizeof(glm::dvec3) * solverData.numVerts);
}

IPCSolver::~IPCSolver()
{
    cudaFree(p);
    cudaFree(xTmp);
    cudaFree(x_n);
}

void IPCSolver::Update(SolverData<double>& solverData, SolverParams& solverParams)
{
    SolverStep(solverData, solverParams);
    solverData.pFixedBodies->HandleCollisions(solverData.X, solverData.V, solverData.numVerts, (double)solverParams.muT, (double)solverParams.muN);
}

void IPCSolver::SolverPrepare(SolverData<double>& solverData, SolverParams& solverParams)
{
}
namespace IPC {

    __global__ void computeXTilde(glm::dvec3* xTilde, const glm::dvec3* x, glm::dvec3* v, double dt, int numVerts)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= numVerts) return;

        xTilde[idx] = x[idx] + dt * v[idx];
    }
    __global__ void computeXMinusAP(glm::dvec3* xPlusAP, const glm::dvec3* x, const double* p, double alpha, int numVerts)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= numVerts) return;

        xPlusAP[idx].x = x[idx].x - alpha * p[idx * 3];
        xPlusAP[idx].y = x[idx].y - alpha * p[idx * 3 + 1];
        xPlusAP[idx].z = x[idx].z - alpha * p[idx * 3 + 2];
    }

    __global__ void updateVel(const glm::dvec3* x, const glm::dvec3* x_n, glm::dvec3* v, double dtInv, int numVerts)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numVerts)
        {
            v[index] = (x[index] - x_n[index]) * dtInv;
        }
    }
}

void IPCSolver::SolverStep(SolverData<double>& solverData, SolverParams& solverParams)
{
    double h = solverParams.dt;
    double h2 = h * h;
    int blocks = (solverData.numVerts + threadsPerBlock - 1) / threadsPerBlock;
    cudaMemcpy(x_n, solverData.X, sizeof(glm::dvec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
    IPC::computeXTilde << <blocks, threadsPerBlock >> > (solverData.XTilde, solverData.X, solverData.V, h, solverData.numVerts);

    double E_last = 0;
    E_last = energy.Val(solverData.X, solverData, h2);

    SearchDirection(solverData, h2);
    while (!EndCondition(h)) {
        double alpha = 1;
        while (true) {
            IPC::computeXMinusAP << <blocks, threadsPerBlock >> > (xTmp, solverData.X, p, alpha, solverData.numVerts);
            double E = energy.Val(xTmp, solverData, h2);
            if (E > E_last) {
                alpha /= 2;
            }
            else {
                break;
            }
        }
        cudaMemcpy(solverData.X, xTmp, sizeof(glm::dvec3) * solverData.numVerts, cudaMemcpyDeviceToDevice);
        E_last = energy.Val(solverData.X, solverData, h2);
        SearchDirection(solverData, h2);
    }
    IPC::updateVel << <blocks, threadsPerBlock >> > (solverData.X, x_n, solverData.V, 1.0 / h, solverData.numVerts);
}

void IPCSolver::SearchDirection(SolverData<double>& solverData, double h2)
{
    energy.Gradient(solverData, h2);
    energy.Hessian(solverData, h2);
    linearSolver->Solve(solverData.numVerts * 3, energy.gradient, p, energy.hessianVal, energy.nnz, energy.hessianRowIdx, energy.hessianColIdx);
}

bool IPCSolver::EndCondition(double h)
{
    thrust::device_ptr<double> dev_ptr(p);
    double inf_norm = thrust::transform_reduce(dev_ptr, dev_ptr + numVerts * 3,
        [] __host__ __device__(double x) { return abs(x); }, 0.0, thrust::maximum<double>()) / h;

    return inf_norm < tolerance;
}

IPEnergy::IPEnergy(const SolverData<double>& solverData) : inertia(solverData, nnz, solverData.numVerts, solverData.mass),
elastic(new CorotatedEnergy<double>(solverData, nnz))
{
    cudaMalloc(&gradient, sizeof(double) * solverData.numVerts * 3);
    cudaMalloc(&hessianVal, sizeof(double) * nnz);
    cudaMalloc(&hessianRowIdx, sizeof(int) * nnz);
    cudaMalloc(&hessianColIdx, sizeof(int) * nnz);
    inertia.SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
    elastic->SetHessianPtr(hessianVal, hessianRowIdx, hessianColIdx);
}

IPEnergy::~IPEnergy()
{
    cudaFree(gradient);
    cudaFree(hessianVal);
    cudaFree(hessianRowIdx);
    cudaFree(hessianColIdx);
}

double IPEnergy::Val(const glm::dvec3* Xs, const SolverData<double>& solverData, double h2) const
{
    // double inertiaEnergy = inertia.Val(Xs, solverData);
    // double gravityEnergy = gravity.Val(Xs, solverData);
    // double elasticEnergy = elastic->Val(Xs, solverData);
    return inertia.Val(Xs, solverData) + h2 * (gravity.Val(Xs, solverData) + elastic->Val(Xs, solverData));
}

void IPEnergy::Gradient(const SolverData<double>& solverData, double h2) const
{
    cudaMemset(gradient, 0, sizeof(double) * solverData.numVerts * 3);
    inertia.Gradient(gradient, solverData, 1);
    gravity.Gradient(gradient, solverData, h2);
    elastic->Gradient(gradient, solverData, h2);
}

void IPEnergy::Hessian(const SolverData<double>& solverData, double h2) const
{
    inertia.Hessian(solverData, 1);
    gravity.Hessian(solverData, h2);
    elastic->Hessian(solverData, h2);
}