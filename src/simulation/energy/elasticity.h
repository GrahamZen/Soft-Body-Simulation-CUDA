#pragma once
#include <energy/energy.h>

template <typename HighP>
class ElasticEnergy : public Energy<HighP> {
public:
    ElasticEnergy(int& hessianIdxOffset);
    virtual ~ElasticEnergy() = default;
    virtual HighP Val(const SolverData<HighP>& solverData, HighP coef) const = 0;
    virtual void Gradient(HighP* grad, const SolverData<HighP>& solverData, HighP coef) const = 0;
    virtual void Hessian(const SolverData<HighP>& solverData, HighP coef) const = 0;
};

template<typename HighP>
inline ElasticEnergy<HighP>::ElasticEnergy(int& hessianIdxOffset) :
    Energy<HighP>(hessianIdxOffset)
{
}
