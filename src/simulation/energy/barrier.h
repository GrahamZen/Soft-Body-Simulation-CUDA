#pragma once

#include <energy/energy.h>

template <typename Scalar>
class BarrierEnergy : public Energy<Scalar> {
public:
    BarrierEnergy(const SolverData<Scalar>& solverData, int& hessianIdxOffset, Scalar dHat);
    virtual ~BarrierEnergy() = default;
    virtual int NNZ(const SolverData<Scalar>& solverData) const override;
    virtual Scalar Val(const glm::tvec3<Scalar>* Xs, const SolverData<Scalar>& solverData) const override;
    virtual void Gradient(Scalar* grad, const SolverData<Scalar>& solverData, Scalar coef) const override;
    virtual void Hessian(const SolverData<Scalar>& solverData, Scalar coef) const override;
    Scalar InitStepSize(const SolverData<Scalar>& solverData, Scalar* p) const;
private:
    Scalar dhat = 1e-2;
};
