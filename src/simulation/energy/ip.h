#pragma once

#include <def.h>
#include <energy/inertia.h>
#include <energy/gravity.h>
#include <energy/elasticity.h>
#include <energy/implicitBarrier.h>

class IPEnergy {
public:
    IPEnergy(const SolverData<double>& solverData);
    ~IPEnergy();
    double Val(const glm::dvec3* Xs, const SolverData<double>& solverData, const SolverParams<double>& solverParams, double h2) const;
    void GradientHessian(const SolverData<double>& solverData, const SolverParams<double>& solverParams, double h2) const;
    double InitStepSize(SolverData<double>& solverData, const SolverParams<double>& solverParams, double* p, glm::tvec3<double>* XTmp) const;
    int NNZ(const SolverData<double>& solverData) const;
    double* gradient = nullptr;
    // collision queries should be updated if dirty
    double* hessianVal = nullptr;
    int* hessianRowIdx = nullptr;
    int* hessianColIdx = nullptr;
private:
    int nnz = 0;
    InertiaEnergy<double> inertia;
    GravityEnergy<double> gravity;
    ImplicitBarrierEnergy<double> implicitBarrier;
    ElasticEnergy<double>* elastic = nullptr;
    BarrierEnergy<double> barrier;
};