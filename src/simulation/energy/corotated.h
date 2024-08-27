#pragma once
#include <energy/elasticity.h>
#include <solver/solverUtil.cuh>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

template <typename HighP>
class CorotatedEnergy : public ElasticEnergy<HighP> {
public:
    CorotatedEnergy(const SolverData<HighP>& solverData, int& hessianIdxOffset);
    virtual ~CorotatedEnergy() override = default;
    virtual int NNZ(const SolverData<HighP>& solverData) const override;
    virtual HighP Val(const SolverData<HighP>& solverData) const override;
    virtual void Gradient(HighP* grad, const SolverData<HighP>& solverData) const override;
    virtual void Hessian(const SolverData<HighP>& solverData) const override;
};

namespace Corotated {
    template <typename HighP>
    __global__ void GradientKern(HighP* grad, const glm::tvec3<HighP>* X, const indexType* Tet, const glm::tmat3x3<HighP>* inv_Dm,
        HighP* mu, HighP* lambda, const indexType numTets) {
        indexType tetIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (tetIndex >= numTets) return;

        glm::tmat3x3<HighP> Ds = Build_Edge_Matrix(X, Tet, tetIndex);
        glm::tmat3x3<HighP> F = Ds * inv_Dm[tetIndex];
        glm::tmat3x3<HighP> U, S, V;

        svdGLM(F, U, S, V);
        glm::tmat3x3<HighP> R = U * glm::transpose(V);
        glm::tmat3x3<HighP> P = 2 * mu[tetIndex] * (F - R) + lambda[tetIndex] * trace(glm::transpose(R) * F - glm::tmat3x3<HighP>(1)) * R;
        glm::tmat3x3<HighP> dPsidx = P * inv_Dm[tetIndex];
        atomicAdd(&grad[Tet[tetIndex * 4 + 0] * 3 + 0], dPsidx[0][0]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 0] * 3 + 1], dPsidx[0][1]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 0] * 3 + 2], dPsidx[0][2]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 1] * 3 + 0], dPsidx[1][0]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 1] * 3 + 1], dPsidx[1][1]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 1] * 3 + 2], dPsidx[1][2]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 2] * 3 + 0], dPsidx[2][0]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 2] * 3 + 1], dPsidx[2][1]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 2] * 3 + 2], dPsidx[2][2]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 3] * 3 + 0], -dPsidx[0][0] - dPsidx[1][0] - dPsidx[2][0]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 3] * 3 + 1], -dPsidx[0][1] - dPsidx[1][1] - dPsidx[2][1]);
        atomicAdd(&grad[Tet[tetIndex * 4 + 3] * 3 + 2], -dPsidx[0][2] - dPsidx[1][2] - dPsidx[2][2]);
    }
    template <typename HighP>
    __global__ void HessianKern(HighP* hessianVal, int* hessianRowIdx, int* hessianColIdx, const glm::tvec3<HighP>* X, const indexType* Tet,
        const glm::tmat3x3<HighP>* inv_Dm, HighP* mus, HighP* lambdas, const indexType numTets) {
        indexType tetIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (tetIndex >= numTets) return;
        HighP mu = mus[tetIndex];
        HighP lambda = lambdas[tetIndex];
        glm::tmat3x3<HighP> Ds = Build_Edge_Matrix(X, Tet, tetIndex);
        glm::tmat3x3<HighP> DmInv = inv_Dm[tetIndex];
        glm::tmat3x3<HighP> F = Ds * DmInv;
        glm::tmat3x3<HighP> R, S, U, Sigma, V;
        svdRV(F, U, S, V);
        R = U * glm::transpose(V);
        S = V * S * glm::transpose(V);
        Vector9<HighP> g1(R);
        glm::tmat3x3<HighP> T0(0, -1, 0, 1, 0, 0, 0, 0, 0);
        glm::tmat3x3<HighP> T1(0, 0, 0, 0, 0, 1, 0, -1, 0);
        glm::tmat3x3<HighP> T2(0, 0, 1, 0, 0, 0, -1, 0, 0);
        T0 = 1 / sqrt(2) * U * T0 * glm::transpose(V);
        T1 = 1 / sqrt(2) * U * T1 * glm::transpose(V);
        T2 = 1 / sqrt(2) * U * T2 * glm::transpose(V);
        Vector9<HighP> t0(T0);
        Vector9<HighP> t1(T1);
        Vector9<HighP> t2(T2);
        HighP s0 = S[0][0];
        HighP s1 = S[1][1];
        HighP s2 = S[2][2];
        HighP lambdaI1Minus2muMinus3lambda = (lambda * trace(S) - 2 * mu - 3 * lambda) * 2;
        Matrix9<HighP> d2PsidF2(2 * mu);
        d2PsidF2 += lambda * Matrix9<HighP>(g1, g1);
        d2PsidF2 += (lambdaI1Minus2muMinus3lambda / (s0 + s1)) * (Matrix9<HighP>(t0, t0));
        d2PsidF2 += (lambdaI1Minus2muMinus3lambda / (s1 + s2)) * (Matrix9<HighP>(t1, t1));
        d2PsidF2 += (lambdaI1Minus2muMinus3lambda / (s0 + s2)) * (Matrix9<HighP>(t2, t2));
        Matrix9x12<HighP> PFPx = ComputePFPx(DmInv);
        Matrix12<HighP> Hessian = PFPx.transpose() * d2PsidF2 * PFPx;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        int row = Tet[tetIndex * 4 + i] * 3 + k;
                        int col = Tet[tetIndex * 4 + j] * 3 + l;
                        int idx = (i * 4 + j) * 9 + k * 3 + l;
                        hessianVal[tetIndex * 144 + idx] = Hessian[i * 3 + k][j * 3 + l];
                        hessianRowIdx[tetIndex * 144 + idx] = row;
                        hessianColIdx[tetIndex * 144 + idx] = col;
                    }
                }
            }
        }
    }
}


template<typename HighP>
inline CorotatedEnergy<HighP>::CorotatedEnergy(const SolverData<HighP>& solverData, int& hessianIdxOffset) :
    ElasticEnergy<HighP>(hessianIdxOffset)
{
    hessianIdxOffset += NNZ(solverData);
}

template<typename HighP>
inline int CorotatedEnergy<HighP>::NNZ(const SolverData<HighP>& solverData) const
{
    return solverData.numTets * 144;
}

template <typename HighP>
HighP CorotatedEnergy<HighP>::Val(const SolverData<HighP>& solverData) const {
    HighP sum = thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(solverData.numTets),
        [=] __host__ __device__(indexType tetIndex) {
        glm::tmat3x3<HighP> Ds = Build_Edge_Matrix(solverData.X, solverData.Tet, tetIndex);
        glm::tmat3x3<HighP> V = Ds * solverData.inv_Dm[tetIndex];
        glm::tmat3x3<HighP> U;
        glm::tmat3x3<HighP> S;

        svdGLM(V, U, S, V);
        HighP traceSigmaMinusI = trace(S - glm::tmat3x3<HighP>(1));
        return solverData.mu[tetIndex] * frobeniusNorm(S - glm::tmat3x3<HighP>(1)) * 0.5
            + 0.5 * solverData.lambda[tetIndex] * traceSigmaMinusI * traceSigmaMinusI;
    },
        (HighP)0,
        thrust::plus<HighP>()
    );
    return sum;
}

template <typename HighP>
void CorotatedEnergy<HighP>::Gradient(HighP* grad, const SolverData<HighP>& solverData) const {
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    Corotated::GradientKern << <numBlocks, threadsPerBlock >> > (grad, solverData.X, solverData.Tet, solverData.inv_Dm,
        solverData.mu, solverData.lambda, solverData.numTets);
}

template <typename HighP>
void CorotatedEnergy<HighP>::Hessian(const SolverData<HighP>& solverData) const {
}