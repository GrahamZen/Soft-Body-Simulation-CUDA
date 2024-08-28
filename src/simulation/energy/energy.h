#pragma once
#include <def.h>

template <typename HighP>
class Energy {
public:
    Energy(int hessianIdxOffset);
    Energy() = default;
    virtual ~Energy() = default;
    virtual int NNZ(const SolverData<HighP>& solverData) const = 0;
    virtual HighP Val(const glm::tvec3<HighP>* Xs, const SolverData<HighP>& solverData, HighP coef = 1) const = 0;
    virtual void Gradient(HighP* grad, const SolverData<HighP>& solverData, HighP coef = 1) const = 0;
    virtual void Hessian(const SolverData<HighP>& solverData, HighP coef = 1) const = 0;
    void SetHessianPtr(HighP* hessianVal, int* hessianRowIdx, int* hessianColIdx);
protected:
    HighP* hessianVal = nullptr;
    int* hessianRowIdx = nullptr;
    int* hessianColIdx = nullptr;
    int hessianIdxOffset = -1;
};

template<typename HighP>
inline Energy<HighP>::Energy(int hessianIdxOffset) :hessianIdxOffset(hessianIdxOffset)
{
}

template<typename HighP>
inline void Energy<HighP>::SetHessianPtr(HighP* hessianVal, int* hessianRowIdx, int* hessianColIdx)
{
    if (hessianIdxOffset == -1)return;
    this->hessianVal = hessianVal + hessianIdxOffset;
    this->hessianRowIdx = hessianRowIdx + hessianIdxOffset;
    this->hessianColIdx = hessianColIdx + hessianIdxOffset;
}
