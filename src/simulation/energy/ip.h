#pragma once

#include <def.h>
#include <energy/inertia.h>
#include <energy/gravity.h>
#include <energy/elasticity.h>

struct IPEnergy {
    IPEnergy(const SolverData<double>& solverData);
    ~IPEnergy();
    double Val(const glm::dvec3* Xs, const SolverData<double>& solverData, double h2) const;
    void Gradient(const SolverData<double>& solverData, double h2) const;
    void Hessian(const SolverData<double>& solverData, double h2) const;
    double* gradient = nullptr;
    int nnz = 0;
    double* hessianVal = nullptr;
    int* hessianRowIdx = nullptr;
    int* hessianColIdx = nullptr;
    InertiaEnergy<double> inertia;
    GravityEnergy<double> gravity;
    ElasticEnergy<double>* elastic = nullptr;
};