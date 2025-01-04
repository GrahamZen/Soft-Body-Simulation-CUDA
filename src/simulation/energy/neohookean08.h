#pragma once
#include <energy/elasticity.h>

template <typename Scalar>
class NeoHookean08Energy : public ElasticEnergy<Scalar> {
public:
    NeoHookean08Energy(const SolverData<Scalar>& solverData, int& hessianIdxOffset);
    virtual ~NeoHookean08Energy() override = default;
    virtual int NNZ(const SolverData<Scalar>& solverData) const override;
    virtual Scalar Val(const glm::tvec3<Scalar>* Xs, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams) const override;
    virtual void Gradient(Scalar* grad, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef) const override;
    virtual void Hessian(const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef) const override;
};
