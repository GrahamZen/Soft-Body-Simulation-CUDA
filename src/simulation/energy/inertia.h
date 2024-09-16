#pragma once

#include <energy/energy.h>

template <typename Scalar>
class InertiaEnergy : public Energy<Scalar> {
public:
    InertiaEnergy(const SolverData<Scalar>& solverData, int& hessianIdxOffset, int numVerts, const Scalar* dev_mass);
    virtual ~InertiaEnergy() = default;
    virtual int NNZ(const SolverData<Scalar>& solverData) const override;
    virtual Scalar Val(const glm::tvec3<Scalar>* Xs, const SolverData<Scalar>& solverData) const override;
    virtual void Gradient(Scalar* grad, const SolverData<Scalar>& solverData, Scalar coef) const override;
    virtual void Hessian(const SolverData<Scalar>& solverData, Scalar coef) const override;
};