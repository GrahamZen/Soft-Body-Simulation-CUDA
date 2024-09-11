#pragma once

#include <def.h>
#include <energy/inertia.h>
#include <energy/gravity.h>
#include <energy/elasticity.h>
#include <energy/implicitBarrier.h>

class IPEnergy {
public:
    IPEnergy(const SolverData<double>& solverData, double dHat = 1e-2);
    ~IPEnergy();
    double Val(const glm::dvec3* Xs, const SolverData<double>& solverData, double h2) const;
    void Gradient(const SolverData<double>& solverData, double h2) const;
    void Hessian(const SolverData<double>& solverData, double h2) const;
    double InitStepSize(SolverData<double>& solverData, double* p, glm::dvec3* XTmp) const;
    void UpdateQueries(CollisionDetection<double>* cd, int numVerts, int numTris, const indexType* Tri, const glm::tvec3<double>* X, const indexType* TriFathers, Query*& queries, int& _numQueries);
    double* gradient = nullptr;
    int nnz = 0;
    // collision queries should be updated if dirty
    double* hessianVal = nullptr;
    int* hessianRowIdx = nullptr;
    int* hessianColIdx = nullptr;
private:
    InertiaEnergy<double> inertia;
    GravityEnergy<double> gravity;
    ImplicitBarrierEnergy<double> implicitBarrier;
    ElasticEnergy<double>* elastic = nullptr;
    BarrierEnergy<double> barrier;
};