#pragma once

#include <def.h>

template <typename HighP>
class ElasticEnergy {
public:
    ElasticEnergy() = default;
    virtual ~ElasticEnergy() = default;
    virtual HighP Val(const SolverData<HighP>& solverData) const = 0;
    virtual void Gradient(HighP* grad, const SolverData<HighP>& solverData) const = 0;
    virtual void Hessian(HighP*& hessianVal, int*& hessianRowIdx, int*& hessianColIdx, const SolverData<HighP>& solverData) const = 0;
private:
    HighP* hessianVal = nullptr;
    int* hessianRowIdx = nullptr;
    int* hessianColIdx = nullptr;
};