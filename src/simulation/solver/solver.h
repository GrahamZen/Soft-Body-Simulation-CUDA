#pragma once

#include <def.h>
#include <string>

struct SoftBodyData {
    indexType* Tri = nullptr;
    int numTris = 0;
};

template<typename Scalar>
class Solver {
public:
    Solver(int threadsPerBlock);
    virtual ~Solver();

    virtual void Update(SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams) = 0;
    virtual void Reset();
    void SetPerf(bool val) { perf = val; }
    const std::vector<std::pair<std::string, Scalar>>& GetPerformanceData() const { return performanceData; }
protected:
    virtual void SolverPrepare(SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams) = 0;
    virtual bool SolverStep(SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams) = 0;
    int threadsPerBlock;
    std::vector<std::pair<std::string, Scalar>> performanceData;
    bool perf = false;
    bool solverReady = false;
};

template<typename Scalar>
Solver<Scalar>::Solver(int threadsPerBlock) : threadsPerBlock(threadsPerBlock)
{
}

template<typename Scalar>
Solver<Scalar>::~Solver() {
}

template<typename Scalar>
inline void Solver<Scalar>::Reset()
{
    solverReady = false;
}