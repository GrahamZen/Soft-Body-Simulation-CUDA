#include <energy/neohookean08.h>
#include <solverUtil.cuh>
#include <matrix.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <svd.cuh>

namespace NeoHookean08 {
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
        glm::tmat3x3<Scalar> PJPF;
        PJPF[0] = glm::cross(F[1], F[2]);
        PJPF[1] = glm::cross(F[2], F[0]);
        PJPF[2] = glm::cross(F[0], F[1]);
        Scalar J = glm::determinant(F);
        Scalar invJ = 1 / J;

        glm::tmat3x3<Scalar> P = mu[index] * (F - invJ * PJPF) + lambda[index] * log(J) * invJ * PJPF;
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
        Scalar I_3 = glm::determinant(F);
        Scalar I_3_inv = 1 / I_3;
        Scalar logI_3 = log(I_3);
        Vector9<Scalar> g3;
        glm::tvec3<Scalar>crossF = glm::cross(F[1], F[2]);
        g3[0] = crossF[0]; g3[1] = crossF[1]; g3[2] = crossF[2];
        crossF = glm::cross(F[2], F[0]);
        g3[3] = crossF[0]; g3[4] = crossF[1]; g3[5] = crossF[2];
        crossF = glm::cross(F[0], F[1]);
        g3[6] = crossF[0]; g3[7] = crossF[1]; g3[8] = crossF[2];
        Matrix9<Scalar> H3;
        H3[0][4] = F[2][2]; H3[0][5] = -F[2][1]; H3[0][7] = -F[1][2]; H3[0][8] = F[1][1];
        H3[1][3] = -F[2][2]; H3[1][5] = F[2][0]; H3[1][6] = F[1][2]; H3[1][8] = -F[1][0];
        H3[2][3] = F[2][1]; H3[2][4] = -F[2][0]; H3[2][6] = -F[1][1]; H3[2][7] = F[1][0];
        H3[3][1] = -F[2][2]; H3[3][2] = F[2][1]; H3[3][6] = F[0][2]; H3[3][8] = -F[0][1];
        H3[4][0] = F[2][2]; H3[4][2] = -F[2][0]; H3[4][5] = -F[0][2]; H3[4][8] = F[0][0];
        H3[5][0] = -F[2][1]; H3[5][1] = F[2][0]; H3[5][4] = F[0][1]; H3[5][7] = -F[0][0];
        H3[6][1] = F[1][2]; H3[6][2] = -F[1][1]; H3[6][3] = -F[0][2]; H3[6][7] = F[0][1];
        H3[7][0] = -F[1][2]; H3[7][2] = F[1][0]; H3[7][3] = F[0][1]; H3[7][6] = -F[0][0];
        H3[8][0] = F[1][1]; H3[8][1] = -F[1][0]; H3[8][4] = -F[0][1]; H3[8][5] = F[0][0];
        H3 *= I_3_inv * (lambda * logI_3 - mu);

        Matrix9<Scalar> d2PsidF2(mu);
        d2PsidF2 += ((lambda * (1 - logI_3) + mu) * I_3_inv * I_3_inv) * Matrix9<Scalar>(g3, g3);
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
    template <typename Scalar>
    __global__ void GradHessianKern(Scalar* grad, Scalar* hessianVal, int* hessianRowIdx, int* hessianColIdx, const glm::tvec3<Scalar>* X, const indexType* Tet,
        const glm::tmat3x3<Scalar>* DmInvs, Scalar* Vol, Scalar* mus, Scalar* lambdas, const indexType numTets, Scalar coef) {
        indexType tetIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (tetIndex >= numTets) return;
        const int v0Ind = Tet[tetIndex * 4 + 0] * 3;
        const int v1Ind = Tet[tetIndex * 4 + 1] * 3;
        const int v2Ind = Tet[tetIndex * 4 + 2] * 3;
        const int v3Ind = Tet[tetIndex * 4 + 3] * 3;
        Scalar mu = mus[tetIndex];
        Scalar lambda = lambdas[tetIndex];
        glm::tmat3x3<Scalar> Ds = Build_Edge_Matrix(X, Tet, tetIndex);
        glm::tmat3x3<Scalar> DmInv = DmInvs[tetIndex];
        glm::tmat3x3<Scalar> F = Ds * DmInv;
        Scalar I_3 = glm::determinant(F);
        Scalar I_3_inv = 1 / I_3;
        Scalar logI_3 = log(I_3);
        Scalar V0 = Vol[tetIndex];
        
        glm::tmat3x3<Scalar> PJPF;
        PJPF[0] = glm::cross(F[1], F[2]);
        PJPF[1] = glm::cross(F[2], F[0]);
        PJPF[2] = glm::cross(F[0], F[1]);

        glm::tmat3x3<Scalar> P = mu * (F - I_3_inv * PJPF) + lambda * log(I_3) * I_3_inv * PJPF;
        glm::tmat3x3<Scalar> dPsidx = coef * Vol[tetIndex] * P * glm::transpose(DmInv);
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

        Vector9<Scalar> g3;
        glm::tvec3<Scalar>crossF = glm::cross(F[1], F[2]);
        g3[0] = crossF[0]; g3[1] = crossF[1]; g3[2] = crossF[2];
        crossF = glm::cross(F[2], F[0]);
        g3[3] = crossF[0]; g3[4] = crossF[1]; g3[5] = crossF[2];
        crossF = glm::cross(F[0], F[1]);
        g3[6] = crossF[0]; g3[7] = crossF[1]; g3[8] = crossF[2];
        Matrix9<Scalar> H3;
        H3[0][4] = F[2][2]; H3[0][5] = -F[2][1]; H3[0][7] = -F[1][2]; H3[0][8] = F[1][1];
        H3[1][3] = -F[2][2]; H3[1][5] = F[2][0]; H3[1][6] = F[1][2]; H3[1][8] = -F[1][0];
        H3[2][3] = F[2][1]; H3[2][4] = -F[2][0]; H3[2][6] = -F[1][1]; H3[2][7] = F[1][0];
        H3[3][1] = -F[2][2]; H3[3][2] = F[2][1]; H3[3][6] = F[0][2]; H3[3][8] = -F[0][1];
        H3[4][0] = F[2][2]; H3[4][2] = -F[2][0]; H3[4][5] = -F[0][2]; H3[4][8] = F[0][0];
        H3[5][0] = -F[2][1]; H3[5][1] = F[2][0]; H3[5][4] = F[0][1]; H3[5][7] = -F[0][0];
        H3[6][1] = F[1][2]; H3[6][2] = -F[1][1]; H3[6][3] = -F[0][2]; H3[6][7] = F[0][1];
        H3[7][0] = -F[1][2]; H3[7][2] = F[1][0]; H3[7][3] = F[0][1]; H3[7][6] = -F[0][0];
        H3[8][0] = F[1][1]; H3[8][1] = -F[1][0]; H3[8][4] = -F[0][1]; H3[8][5] = F[0][0];
        H3 *= I_3_inv * (lambda * logI_3 - mu);

        Matrix9<Scalar> d2PsidF2(mu);
        d2PsidF2 += ((lambda * (1 - logI_3) + mu) * I_3_inv * I_3_inv) * Matrix9<Scalar>(g3, g3);
        d2PsidF2 *= (V0 * coef);
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
inline NeoHookean08Energy<Scalar>::NeoHookean08Energy(const SolverData<Scalar>& solverData, int& hessianIdxOffset) :
    ElasticEnergy<Scalar>(hessianIdxOffset)
{
    hessianIdxOffset += NNZ(solverData);
}

template<typename Scalar>
inline int NeoHookean08Energy<Scalar>::NNZ(const SolverData<Scalar>& solverData) const
{
    return solverData.numTets * 144;
}

template <typename Scalar>
Scalar NeoHookean08Energy<Scalar>::Val(const glm::tvec3<Scalar>* Xs, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams) const {
    Scalar sum = thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(solverData.numTets),
        [=] __host__ __device__(indexType tetIndex) {
        glm::tmat3x3<Scalar> Ds = Build_Edge_Matrix(Xs, solverData.Tet, tetIndex);
        glm::tmat3x3<Scalar> F = Ds * solverData.DmInv[tetIndex];
        Scalar Vol = solverData.V0[tetIndex];
        Scalar mu = solverData.mu[tetIndex];
        Scalar logJ = log(glm::determinant(F));
        return Vol * (mu * 0.5 * (frobeniusNorm(F) - 3) - mu * logJ + 0.5 * solverData.lambda[tetIndex] * logJ * logJ);
    },
        (Scalar)0,
        thrust::plus<Scalar>()
    );
    return sum;
}

template <typename Scalar>
void NeoHookean08Energy<Scalar>::Gradient(Scalar* grad, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef) const {
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    NeoHookean08::GradientKern << <numBlocks, threadsPerBlock >> > (grad, solverData.X, solverData.Tet, solverData.DmInv,
        solverData.V0, solverData.mu, solverData.lambda, solverData.numTets, coef);
}

template <typename Scalar>
void NeoHookean08Energy<Scalar>::Hessian(const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef) const {
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    NeoHookean08::HessianKern << <numBlocks, threadsPerBlock >> > (hessianVal, hessianRowIdx, hessianColIdx,
        solverData.X, solverData.Tet, solverData.DmInv, solverData.V0, solverData.mu, solverData.lambda, solverData.numTets, coef);
}

template<typename Scalar>
void NeoHookean08Energy<Scalar>::GradientHessian(Scalar* grad, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef) const
{
    int threadsPerBlock = 256;
    int numBlocks = (solverData.numTets + threadsPerBlock - 1) / threadsPerBlock;
    NeoHookean08::GradHessianKern << <numBlocks, threadsPerBlock >> > (grad, hessianVal, hessianRowIdx, hessianColIdx,
        solverData.X, solverData.Tet, solverData.DmInv, solverData.V0, solverData.mu, solverData.lambda, solverData.numTets, coef);
}

template class NeoHookean08Energy<float>;
template class NeoHookean08Energy<double>;