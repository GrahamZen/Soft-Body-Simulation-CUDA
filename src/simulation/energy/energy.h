#pragma once
#include <def.h>

template <typename Scalar>
class Energy {
public:
    Energy(int hessianIdxOffset);
    Energy() = default;
    virtual ~Energy() = default;
    virtual int NNZ(const SolverData<Scalar>& solverData) const = 0;
    virtual Scalar Val(const glm::tvec3<Scalar>* Xs, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams) const = 0;
    virtual void Gradient(Scalar* grad, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef = 1) const = 0;
    virtual void Hessian(const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef = 1) const = 0;
    virtual void GradientHessian(Scalar* grad, const SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams, Scalar coef = 1) const = 0;
    void SetHessianPtr(Scalar* hessianVal, int* hessianRowIdx, int* hessianColIdx);
protected:
    Scalar* hessianVal = nullptr;
    int* hessianRowIdx = nullptr;
    int* hessianColIdx = nullptr;
    int hessianIdxOffset = -1;
};