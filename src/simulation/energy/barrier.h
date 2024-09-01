#pragma once

#include <energy/energy.h>

template <typename HighP>
class BarrierEnergy : public Energy<HighP> {
public:
    BarrierEnergy(const SolverData<HighP>& solverData, int& hessianIdxOffset, HighP dHat);
    virtual ~BarrierEnergy() = default;
    virtual int NNZ(const SolverData<HighP>& solverData) const override;
    virtual HighP Val(const glm::tvec3<HighP>* Xs, const SolverData<HighP>& solverData) const override;
    virtual void Gradient(HighP* grad, const SolverData<HighP>& solverData, HighP coef) const override;
    virtual void Hessian(const SolverData<HighP>& solverData, HighP coef) const override;
    HighP InitStepSize(const SolverData<HighP>& solverData, HighP* p) const;
private:
    HighP dhat = 1e-2;
};
