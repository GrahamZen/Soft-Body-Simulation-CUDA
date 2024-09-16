#pragma once
#include <energy/elasticity.h>

template <typename Scalar>
class CorotatedEnergy : public ElasticEnergy<Scalar> {
public:
    CorotatedEnergy(const SolverData<Scalar>& solverData, int& hessianIdxOffset);
    virtual ~CorotatedEnergy() override = default;
    virtual int NNZ(const SolverData<Scalar>& solverData) override;
    virtual Scalar Val(const glm::tvec3<Scalar>* Xs, const SolverData<Scalar>& solverData) const override;
    virtual void Gradient(Scalar* grad, const SolverData<Scalar>& solverData, Scalar coef) const override;
    virtual void Hessian(const SolverData<Scalar>& solverData, Scalar coef) const override;
};
