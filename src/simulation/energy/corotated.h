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
    virtual HighP Val(const glm::tvec3<HighP>* Xs, const SolverData<HighP>& solverData) const override;
    virtual void Gradient(HighP* grad, const SolverData<HighP>& solverData, HighP coef) const override;
    virtual void Hessian(const SolverData<HighP>& solverData, HighP coef) const override;
};

namespace Corotated {
    template <typename HighP>
    __global__ void GradientKern(HighP* grad, const glm::tvec3<HighP>* X, const indexType* Tet, const glm::tmat3x3<HighP>* DmInvs,
        HighP* Vol, HighP* mu, HighP* lambda, const int numTets, HighP coef) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numTets) return;
        const int v0Ind = Tet[index * 4 + 0] * 3;
        const int v1Ind = Tet[index * 4 + 1] * 3;
        const int v2Ind = Tet[index * 4 + 2] * 3;
        const int v3Ind = Tet[index * 4 + 3] * 3;
        glm::tmat3x3<HighP> DmInv = DmInvs[index];
        glm::tmat3x3<HighP> Ds = Build_Edge_Matrix(X, Tet, index);
        glm::tmat3x3<HighP> F = Ds * DmInv;
        glm::tmat3x3<HighP> U, S, V;

        svdRV(F, U, S, V);
        glm::tmat3x3<HighP> R = U * glm::transpose(V);
        glm::tmat3x3<HighP> P = 2 * mu[index] * (F - R) + lambda[index] * trace(glm::transpose(R) * F - glm::tmat3x3<HighP>(1)) * R;
        glm::tmat3x3<HighP> dPsidx = coef * Vol[index] * P * glm::transpose(DmInv);
        atomicAdd(&grad[v0Ind + 0], -dPsidx[0][0] - dPsidx[1][0] - dPsidx[2][0]);
        atomicAdd(&grad[v0Ind + 1], -dPsidx[0][1] - dPsidx[1][1] - dPsidx[2][1]);
        atomicAdd(&grad[v0Ind + 2], -dPsidx[0][2] - dPsidx[1][2] - dPsidx[2][2]);
        atomicAdd(&grad[v1Ind + 0], dPsidx[0][0]);
        atomicAdd(&grad[v1Ind + 1], dPsidx[0][1]);
        atomicAdd(&grad[v1Ind + 2], dPsidx[0][2]);
        atomicAdd(&grad[v2Ind + 0], dPsidx[1][0]);
        atomicAdd(&grad[v2Ind + 1], dPsidx[1][1]);
        atomicAdd(&grad[v2Ind + 2], dPsidx[1][2]);
        atomicAdd(&grad[v3Ind + 0], dPsidx[2][0]);
        atomicAdd(&grad[v3Ind + 1], dPsidx[2][1]);
        atomicAdd(&grad[v3Ind + 2], dPsidx[2][2]);
    }

    template <typename HighP>
    __global__ void HessianKern(HighP* hessianVal, int* hessianRowIdx, int* hessianColIdx, const glm::tvec3<HighP>* X, const indexType* Tet,
        const glm::tmat3x3<HighP>* DmInvs, HighP* Vol, HighP* mus, HighP* lambdas, const indexType numTets, HighP coef) {
        indexType tetIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (tetIndex >= numTets) return;
        HighP mu = mus[tetIndex];
        HighP lambda = lambdas[tetIndex];
        glm::tmat3x3<HighP> Ds = Build_Edge_Matrix(X, Tet, tetIndex);
        glm::tmat3x3<HighP> DmInv = DmInvs[tetIndex];
        glm::tmat3x3<HighP> F = Ds * DmInv;
        glm::tmat3x3<HighP> R, S, U, Sigma, V;
        svdRV(F, U, Sigma, V);
        R = U * glm::transpose(V);
        S = V * Sigma * glm::transpose(V);
        Vector9<HighP> g1(R);
        glm::tmat3x3<HighP> T0(0, -1, 0, 1, 0, 0, 0, 0, 0);
        glm::tmat3x3<HighP> T1(0, 0, 0, 0, 0, 1, 0, -1, 0);
        glm::tmat3x3<HighP> T2(0, 0, 1, 0, 0, 0, -1, 0, 0);
        T0 = 1 / sqrt((HighP)2) * U * T0 * glm::transpose(V);
        T1 = 1 / sqrt((HighP)2) * U * T1 * glm::transpose(V);
        T2 = 1 / sqrt((HighP)2) * U * T2 * glm::transpose(V);
        Vector9<HighP> t0(T0);
        Vector9<HighP> t1(T1);
        Vector9<HighP> t2(T2);
        HighP s0 = Sigma[0][0];
        HighP s1 = Sigma[1][1];
        HighP s2 = Sigma[2][2];
        HighP lambda0 = 2 / (s0 + s1);
        HighP lambda1 = 2 / (s1 + s2);
        HighP lambda2 = 2 / (s0 + s2);
        if (s0 + s1 < 2)
            lambda0 = 1;
        if (s1 + s2 < 2)
            lambda1 = 1;
        if (s0 + s2 < 2)
            lambda2 = 1;
        HighP lambdaI1Minus2muMinus3lambda = lambda * trace(S) - 2 * mu - 3 * lambda;
        Matrix9<HighP> d2PsidF2(2 * mu);
        d2PsidF2 += lambda * Matrix9<HighP>(g1, g1);
        d2PsidF2 += (lambdaI1Minus2muMinus3lambda * lambda0) * (Matrix9<HighP>(t0, t0));
        d2PsidF2 += (lambdaI1Minus2muMinus3lambda * lambda1) * (Matrix9<HighP>(t1, t1));
        d2PsidF2 += (lambdaI1Minus2muMinus3lambda * lambda2) * (Matrix9<HighP>(t2, t2));
        Matrix9x12<HighP> PFPx = ComputePFPx(DmInv);
        Matrix12<HighP> Hessian = Vol[tetIndex] * coef * PFPx.transpose() * d2PsidF2 * PFPx;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        int row = Tet[tetIndex * 4 + i] * 3 + k;
                        int col = Tet[tetIndex * 4 + j] * 3 + l;
                        int idx = tetIndex * 144 + (i * 4 + j) * 9 + k * 3 + l;
                        hessianVal[idx] = Hessian[i * 3 + k][j * 3 + l];
                        hessianRowIdx[idx] = row;
                        hessianColIdx[idx] = col;
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
HighP CorotatedEnergy<HighP>::Val(const glm::tvec3<HighP>* Xs, const SolverData<HighP>& solverData) const {
    HighP sum = thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(solverData.numTets),
        [=] __host__ __device__(indexType tetIndex) {
        glm::tmat3x3<HighP> Ds = Build_Edge_Matrix(Xs, solverData.Tet, tetIndex);
        glm::tmat3x3<HighP> V = Ds * solverData.DmInv[tetIndex];
        glm::tmat3x3<HighP> U;
        glm::tmat3x3<HighP> Sigma;
        HighP Vol = solverData.V0[tetIndex];
        svdGLM(V, U, Sigma, V);
        HighP traceSigmaMinusI = trace(Sigma - glm::tmat3x3<HighP>(1));
        return Vol * (solverData.mu[tetIndex] * frobeniusNorm(Sigma - glm::tmat3x3<HighP>(1))
            + 0.5 * solverData.lambda[tetIndex] * traceSigmaMinusI * traceSigmaMinusI);
    },
        (HighP)0,
        thrust::plus<HighP>()
    );
    return sum;
}

template <typename HighP>
void CorotatedEnergy<HighP>::Gradient(HighP* grad, const SolverData<HighP>& solverData, HighP coef) const {
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    Corotated::GradientKern << <numBlocks, threadsPerBlock >> > (grad, solverData.X, solverData.Tet, solverData.DmInv,
        solverData.V0, solverData.mu, solverData.lambda, solverData.numTets, coef);
}

template <typename HighP>
void CorotatedEnergy<HighP>::Hessian(const SolverData<HighP>& solverData, HighP coef) const {
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    Corotated::HessianKern << <numBlocks, threadsPerBlock >> > (hessianVal, hessianRowIdx, hessianColIdx,
        solverData.X, solverData.Tet, solverData.DmInv, solverData.V0, solverData.mu, solverData.lambda, solverData.numTets, coef);
}