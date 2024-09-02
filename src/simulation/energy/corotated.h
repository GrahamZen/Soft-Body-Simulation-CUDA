#pragma once
#include <energy/elasticity.h>

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
