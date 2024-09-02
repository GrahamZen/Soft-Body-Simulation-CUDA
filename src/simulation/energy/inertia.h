#pragma once

#include <energy/energy.h>

template <typename HighP>
class InertiaEnergy : public Energy<HighP> {
public:
    InertiaEnergy(const SolverData<HighP>& solverData, int& hessianIdxOffset, int numVerts, const HighP* dev_mass);
    virtual ~InertiaEnergy() = default;
    virtual int NNZ(const SolverData<HighP>& solverData) const override;
    virtual HighP Val(const glm::tvec3<HighP>* Xs, const SolverData<HighP>& solverData) const override;
    virtual void Gradient(HighP* grad, const SolverData<HighP>& solverData, HighP coef) const override;
    virtual void Hessian(const SolverData<HighP>& solverData, HighP coef) const override;
};