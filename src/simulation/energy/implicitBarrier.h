#pragma once

#include <energy/barrier.h>

template <typename Scalar>
class ImplicitBarrierEnergy : public Energy<Scalar> {
public:
    ImplicitBarrierEnergy(const SolverData<Scalar>& solverData, int& hessianIdxOffset, Scalar dhat);
    virtual ~ImplicitBarrierEnergy() = default;
    virtual int NNZ(const SolverData<Scalar>& solverData) const override;
    virtual Scalar Val(const glm::tvec3<Scalar>* Xs, const SolverData<Scalar>& solverData) const override;
    virtual void Gradient(Scalar* grad, const SolverData<Scalar>& solverData, Scalar coef) const override;
    virtual void Hessian(const SolverData<Scalar>& solverData, Scalar coef) const override;
    Scalar InitStepSize(const SolverData<Scalar>& solverData, Scalar* p) const;
private:
    const Scalar dhat = 1e-2;
};
