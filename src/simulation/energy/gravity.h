#pragma once

#include <energy/energy.h>

template <typename HighP>
class GravityEnergy : public Energy<HighP> {
public:
    GravityEnergy() = default;
    virtual int NNZ(const SolverData<HighP>& solverData) const override;
    virtual HighP Val(const glm::tvec3<HighP>* Xs, const SolverData<HighP>& solverData) const override;
    virtual void Gradient(HighP* grad, const SolverData<HighP>& solverData, HighP coef) const override;
    virtual void Hessian(const SolverData<HighP>& solverData, HighP coef) const override {}
    const HighP g = 9.8;
};
