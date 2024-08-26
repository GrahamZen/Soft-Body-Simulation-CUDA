#include <energy/elasticity.h>
#include <solver/solverUtil.cuh>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

template <typename HighP>
class CorotatedEnergy : public ElasticEnergy<HighP> {
public:
    CorotatedEnergy() = default;
    virtual ~CorotatedEnergy() override = default;
    virtual HighP Val(const SolverData<HighP>& solverData) const override;
    virtual void Gradient(HighP* grad, const SolverData<HighP>& solverData) const override;
    virtual void Hessian(HighP*& hessianVal, int*& hessianRowIdx, int*& hessianColIdx, const SolverData<HighP>& solverData) const override;
};

namespace Corotated {
    template <typename HighP>
    __device__ HighP frobeniusNorm(const glm::tmat3x3<HighP>& m) {
        return sqrt(m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2] +
            m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2] +
            m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2]);
    }
}


template <typename HighP>
HighP CorotatedEnergy<HighP>::Val(const SolverData<HighP>& solverData) const {
    thrust::device_ptr<glm::tvec3<HighP>> dev_ptr(solverData.dev_x);
    thrust::device_ptr<indexType> dev_tet(solverData.dev_tet);
    thrust::device_ptr<glm::tmat3x3<HighP>> dev_inv_Dm(solverData.dev_inv_Dm);

    HighP sum = thrust::transform_reduce(
        thrust::counting_iterator<indexType>(0),
        thrust::counting_iterator<indexType>(solverData.numTets),
        [=] __device__(indexType tetIndex) {
        glm::tmat3x3<HighP> Ds = Build_Edge_Matrix(solverData.X, solverData.Tet, tetIndex);
        glm::tmat3x3<HighP> V = Ds * dev_inv_Dm[tetIndex];
        glm::tmat3x3<HighP> U;
        glm::tmat3x3<HighP> S;

        svdGLM(V, U, S, V);
        return Corotated::frobeniusNorm(S - glm::tmat3x3<HighP>(1)) * 0.5;
    },
        (HighP)0,
        thrust::plus<HighP>()
    );
    return sum;
}

template <typename HighP>
void CorotatedEnergy<HighP>::Gradient(HighP* grad, const SolverData<HighP>& solverData) const {
}

template <typename HighP>
void CorotatedEnergy<HighP>::Hessian(HighP*& hessianVal, int*& hessianRowIdx, int*& hessianColIdx, const SolverData<HighP>& solverData) const {
}