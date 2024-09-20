#pragma once
#include <energy/energy.h>

template <typename Scalar>
class ElasticEnergy : public Energy<Scalar> {
public:
    ElasticEnergy(int& hessianIdxOffset);
    virtual ~ElasticEnergy() = default;
    virtual Scalar Val(const glm::tvec3<Scalar>* Xs, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams) const = 0;
    virtual void Gradient(Scalar* grad, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef) const = 0;
    virtual void Hessian(const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef) const = 0;
};