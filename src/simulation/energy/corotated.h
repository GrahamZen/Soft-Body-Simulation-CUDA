#include <energy/elasticity.h>
#include <solver/solverUtil.cuh>

template <typename HighP>
class CorotatedEnergy : public ElasticEnergy<HighP> {
public:
    CorotatedEnergy() = default;
    virtual ~CorotatedEnergy() override = default;
    virtual HighP Val(const SolverData<HighP>& solverData) const override;
    virtual void Gradient(HighP* grad, const SolverData<HighP>& solverData) const override;
    virtual void Hessian(HighP*& hessianVal, int*& hessianRowIdx, int*& hessianColIdx, const SolverData<HighP>& solverData) const override;
};

namespace Corotated{
    template <typename HighP>
    __global__ void computeDeformationGrad(const glm::tvec3<HighP>* X, const indexType* Tet, const glm::mat3* inv_Dm, glm::mat3* F, int numTets) {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < numTets)
        {
            glm::mat3 Ds = Build_Edge_Matrix(X, Tet, index);
            F[index] = Ds * inv_Dm[index];
        }
    }
}


template <typename HighP>
HighP CorotatedEnergy<HighP>::Val(const SolverData<HighP>& solverData) const {
    return 0;
}

template <typename HighP>
void CorotatedEnergy<HighP>::Gradient(HighP* grad, const SolverData<HighP>& solverData) const {
}

template <typename HighP>
void CorotatedEnergy<HighP>::Hessian(HighP*& hessianVal, int*& hessianRowIdx, int*& hessianColIdx, const SolverData<HighP>& solverData) const {
}