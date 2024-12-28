#include <energy/corotated.h>
#include <solverUtil.cuh>
#include <matrix.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <svd.cuh>

namespace Corotated {
    template <typename Scalar>
    __global__ void GradientKern(Scalar* grad, const glm::tvec3<Scalar>* X, const indexType* Tet, const glm::tmat3x3<Scalar>* DmInvs,
        Scalar* Vol, Scalar* mu, Scalar* lambda, const int numTets, Scalar coef) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numTets) return;
        const int v0Ind = Tet[index * 4 + 0] * 3;
        const int v1Ind = Tet[index * 4 + 1] * 3;
        const int v2Ind = Tet[index * 4 + 2] * 3;
        const int v3Ind = Tet[index * 4 + 3] * 3;
        glm::tmat3x3<Scalar> DmInv = DmInvs[index];
        glm::tmat3x3<Scalar> Ds = Build_Edge_Matrix(X, Tet, index);
        glm::tmat3x3<Scalar> F = Ds * DmInv;
        glm::tmat3x3<Scalar> U, S, V;

        svdRV(F, U, S, V);
        glm::tmat3x3<Scalar> R = U * glm::transpose(V);
        glm::tmat3x3<Scalar> P = 2 * mu[index] * (F - R) + lambda[index] * trace(glm::transpose(R) * F - glm::tmat3x3<Scalar>(1)) * R;
        glm::tmat3x3<Scalar> dPsidx = coef * Vol[index] * P * glm::transpose(DmInv);
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

    template <typename Scalar>
    __global__ void HessianKern(Scalar* hessianVal, int* hessianRowIdx, int* hessianColIdx, const glm::tvec3<Scalar>* X, const indexType* Tet,
        const glm::tmat3x3<Scalar>* DmInvs, Scalar* Vol, Scalar* mus, Scalar* lambdas, const indexType numTets, Scalar coef) {
        indexType tetIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (tetIndex >= numTets) return;
        Scalar mu = mus[tetIndex];
        Scalar lambda = lambdas[tetIndex];
        glm::tmat3x3<Scalar> Ds = Build_Edge_Matrix(X, Tet, tetIndex);
        glm::tmat3x3<Scalar> DmInv = DmInvs[tetIndex];
        glm::tmat3x3<Scalar> F = Ds * DmInv;
        glm::tmat3x3<Scalar> R, S, U, Sigma, V;
        svdRV(F, U, Sigma, V);
        R = U * glm::transpose(V);
        S = V * Sigma * glm::transpose(V);
        Vector9<Scalar> g1(R);
        glm::tmat3x3<Scalar> T0(0, -1, 0, 1, 0, 0, 0, 0, 0);
        glm::tmat3x3<Scalar> T1(0, 0, 0, 0, 0, 1, 0, -1, 0);
        glm::tmat3x3<Scalar> T2(0, 0, 1, 0, 0, 0, -1, 0, 0);
        T0 = 1 / sqrt((Scalar)2) * U * T0 * glm::transpose(V);
        T1 = 1 / sqrt((Scalar)2) * U * T1 * glm::transpose(V);
        T2 = 1 / sqrt((Scalar)2) * U * T2 * glm::transpose(V);
        Vector9<Scalar> t0(T0);
        Vector9<Scalar> t1(T1);
        Vector9<Scalar> t2(T2);
        Scalar s0 = Sigma[0][0];
        Scalar s1 = Sigma[1][1];
        Scalar s2 = Sigma[2][2];
        Scalar lambda0 = 2 / (s0 + s1);
        Scalar lambda1 = 2 / (s1 + s2);
        Scalar lambda2 = 2 / (s0 + s2);
        if (s0 + s1 < 2)
            lambda0 = 1;
        if (s1 + s2 < 2)
            lambda1 = 1;
        if (s0 + s2 < 2)
            lambda2 = 1;
        Scalar lambdaI1Minus2muMinus3lambda = lambda * trace(S) - 2 * mu - 3 * lambda;
        Matrix9<Scalar> d2PsidF2(2 * mu);
        d2PsidF2 += lambda * Matrix9<Scalar>(g1, g1);
        d2PsidF2 += (lambdaI1Minus2muMinus3lambda * lambda0) * (Matrix9<Scalar>(t0, t0));
        d2PsidF2 += (lambdaI1Minus2muMinus3lambda * lambda1) * (Matrix9<Scalar>(t1, t1));
        d2PsidF2 += (lambdaI1Minus2muMinus3lambda * lambda2) * (Matrix9<Scalar>(t2, t2));
        d2PsidF2 *= (Vol[tetIndex] * coef);
        Matrix12<Scalar> Hessian;
        ComputeHessian(&DmInv[0][0], d2PsidF2, Hessian);
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


template<typename Scalar>
inline CorotatedEnergy<Scalar>::CorotatedEnergy(const SolverData<Scalar>& solverData, int& hessianIdxOffset) :
    ElasticEnergy<Scalar>(hessianIdxOffset)
{
    hessianIdxOffset += NNZ(solverData);
}

template<typename Scalar>
inline int CorotatedEnergy<Scalar>::NNZ(const SolverData<Scalar>& solverData) const
{
    return solverData.numTets * 144;
}

template <typename Scalar>
Scalar CorotatedEnergy<Scalar>::Val(const glm::tvec3<Scalar>* Xs, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams) const {
    Scalar sum = thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(solverData.numTets),
        [=] __host__ __device__(indexType tetIndex) {
        glm::tmat3x3<Scalar> Ds = Build_Edge_Matrix(Xs, solverData.Tet, tetIndex);
        glm::tmat3x3<Scalar> V = Ds * solverData.DmInv[tetIndex];
        glm::tmat3x3<Scalar> U;
        glm::tmat3x3<Scalar> Sigma;
        Scalar Vol = solverData.V0[tetIndex];
        svdGLM(V, U, Sigma, V);
        Scalar traceSigmaMinusI = trace(Sigma - glm::tmat3x3<Scalar>(1));
        return Vol * (solverData.mu[tetIndex] * frobeniusNorm(Sigma - glm::tmat3x3<Scalar>(1))
            + 0.5 * solverData.lambda[tetIndex] * traceSigmaMinusI * traceSigmaMinusI);
    },
        (Scalar)0,
        thrust::plus<Scalar>()
    );
    return sum;
}

template <typename Scalar>
void CorotatedEnergy<Scalar>::Gradient(Scalar* grad, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef) const {
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    Corotated::GradientKern << <numBlocks, threadsPerBlock >> > (grad, solverData.X, solverData.Tet, solverData.DmInv,
        solverData.V0, solverData.mu, solverData.lambda, solverData.numTets, coef);
}

template <typename Scalar>
void CorotatedEnergy<Scalar>::Hessian(const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef) const {
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    Corotated::HessianKern << <numBlocks, threadsPerBlock >> > (hessianVal, hessianRowIdx, hessianColIdx,
        solverData.X, solverData.Tet, solverData.DmInv, solverData.V0, solverData.mu, solverData.lambda, solverData.numTets, coef);
}

template class CorotatedEnergy<float>;
template class CorotatedEnergy<double>;